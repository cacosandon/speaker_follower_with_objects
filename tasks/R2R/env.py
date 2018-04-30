''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('build')
import MatterSim
import csv
import numpy as np
import math
import json
import random
import networkx as nx
import functools
import os.path
import time
import paths
import pickle

from collections import namedtuple, defaultdict

from utils import load_datasets, load_nav_graphs, structured_map, vocab_pad_idx, decode_base64, k_best_indices, try_cuda

import torch
from torch.autograd import Variable

csv.field_size_limit(sys.maxsize)

FOLLOWER_MODEL_ACTIONS = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']

LEFT_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("left")
RIGHT_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("right")
UP_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("up")
DOWN_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("down")
FORWARD_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("forward")
END_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("<end>")
START_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("<start>")
IGNORE_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("<ignore>")


FOLLOWER_ENV_ACTIONS = [
    (0,-1, 0), # left
    (0, 1, 0), # right
    (0, 0, 1), # up
    (0, 0,-1), # down
    (1, 0, 0), # forward
    (0, 0, 0), # <end>
    (0, 0, 0), # <start>
    (0, 0, 0)  # <ignore>
]

assert len(FOLLOWER_MODEL_ACTIONS) == len(FOLLOWER_ENV_ACTIONS)

WorldState = namedtuple("WorldState", ["scanId", "viewpointId", "heading", "elevation"])

BottomUpViewpoint = namedtuple("BottomUpViewpoint", ["cls_prob", "image_features", "attribute_indices", "object_indices", "boxes", "no_object_mask"])

def load_world_state(sim, world_state):
    sim.newEpisode(*world_state)

def get_world_state(sim):
    state = sim.getState()
    return WorldState(scanId=state.scanId,
                      viewpointId=state.location.viewpointId,
                      heading=state.heading,
                      elevation=state.elevation)

def make_sim(image_w, image_h, vfov):
    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)
    sim.setCameraResolution(image_w, image_h)
    sim.setCameraVFOV(math.radians(vfov))
    sim.init()
    return sim

# def encode_action_sequence(action_tuples):
#     encoded = []
#     reached_end = False
#     if action_tuples[0] == (0, 0, 0):
#         # this method can't handle a <start> symbol
#         assert all(t == (0, 0, 0) for t in action_tuples)
#     for tpl in action_tuples:
#         if tpl == (0, 0, 0):
#             if reached_end:
#                 ix = IGNORE_ACTION_INDEX
#             else:
#                 ix = END_ACTION_INDEX
#                 reached_end = True
#         else:
#             ix = FOLLOWER_ENV_ACTIONS.index(tpl)
#         encoded.append(ix)
#     return encoded

def index_action_tuple(action_tuple):
    ix, heading_chg, elevation_chg = action_tuple
    if heading_chg > 0:
        return FOLLOWER_MODEL_ACTIONS.index('right')
    elif heading_chg < 0:
        return FOLLOWER_MODEL_ACTIONS.index('left')
    elif elevation_chg > 0:
        return FOLLOWER_MODEL_ACTIONS.index('up')
    elif elevation_chg < 0:
        return FOLLOWER_MODEL_ACTIONS.index('down')
    elif ix > 0:
        return FOLLOWER_MODEL_ACTIONS.index('forward')
    else:
        return FOLLOWER_MODEL_ACTIONS.index('<end>')

class ImageFeatures(object):
    NUM_VIEWS = 36
    MEAN_POOLED_DIM = 2048

    def __init__(self, image_feature_type, image_feature_datasets, split_convolutional_features=True, downscale_convolutional_features=True):
        self.image_feature_type = image_feature_type
        image_feature_datasets = sorted(image_feature_datasets)
        self.image_feature_datasets = image_feature_datasets
        self.split_convolutional_features = split_convolutional_features
        self.downscale_convolutional_features = downscale_convolutional_features

        self.convolutional_feature_stores = [paths.convolutional_feature_store_paths[dataset]
                                             for dataset in image_feature_datasets]
        self.mean_pooled_feature_stores = [paths.mean_pooled_feature_store_paths[dataset]
                                           for dataset in image_feature_datasets]
        if image_feature_type == 'mean_pooled':
            self.feature_dim = ImageFeatures.MEAN_POOLED_DIM * len(image_feature_datasets)
            print('Loading image features from %s' % ', '.join(self.mean_pooled_feature_stores))
            tsv_fieldnames = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
            self.features = defaultdict(list)
            for mpfs in self.mean_pooled_feature_stores:
                with open(mpfs, "rt") as tsv_in_file:
                    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = tsv_fieldnames)
                    for item in reader:
                        self.image_h = int(item['image_h'])
                        self.image_w = int(item['image_w'])
                        self.vfov = int(item['vfov'])
                        long_id = self._make_id(item['scanId'], item['viewpointId'])
                        features = np.frombuffer(decode_base64(item['features']), dtype=np.float32).reshape((ImageFeatures.NUM_VIEWS, ImageFeatures.MEAN_POOLED_DIM))
                        self.features[long_id].append(features)
            assert all(len(feats) == len(self.mean_pooled_feature_stores) for feats in self.features.values())
            self.features = {
                long_id: np.concatenate(feats, axis=1)
                for long_id, feats in self.features.items()
            }
        else:
            if image_feature_type == 'attention':
                print("Will load convolutional or bottom_up features on demand")
            else:
                print('Image features not provided')
            self.feature_dim = ImageFeatures.MEAN_POOLED_DIM
            self.features = np.zeros(self.feature_dim, dtype=np.float32)
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60

    @staticmethod
    def from_args(args):
        if args.image_feature_type == "attention" and args.image_attention_type == "bottom_up":
            return BottomUpImageFeatures(args.bottom_up_detections, precomputed_cache_path=paths.bottom_up_feature_cache_path)
        else:
            return ImageFeatures(args.image_feature_type, args.image_feature_datasets)

    @staticmethod
    def add_args(argument_parser):
        argument_parser.add_argument("--image_feature_type", choices=["mean_pooled", "none", "attention"], default="mean_pooled")
        argument_parser.add_argument("--image_attention_type", choices=["feedforward", "multiplicative", "bottom_up"], default="feedforward")
        argument_parser.add_argument("--image_attention_size", type=int)
        argument_parser.add_argument("--image_feature_datasets", nargs="+", default=["imagenet"], help="{imagenet,places365}")
        argument_parser.add_argument("--bottom_up_detections", type=int, default=10)
        argument_parser.add_argument("--bottom_up_detection_embedding_size", type=int, default=20)

    @staticmethod
    def get_name(args):
        if args.image_feature_type == "none":
            return args.image_feature_type
        name = '+'.join(sorted(args.image_feature_datasets))
        if args.image_feature_type == "attention":
            name = name + "_" + args.image_attention_type + "_attention"
            if args.image_attention_type == "bottom_up":
                name = name + "_d={}".format(args.bottom_up_detections)
        return name

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def batch_features(self, feature_list):
        features = np.stack(feature_list)
        return try_cuda(Variable(torch.from_numpy(features), requires_grad=False))

    @functools.lru_cache(maxsize=3000)
    def _get_convolutional_features(self, scanId, viewpointId, viewIndex):
        feats = []
        for cfs in self.convolutional_feature_stores:
            if self.split_convolutional_features:
                path = os.path.join(cfs, scanId, "{}_{}{}.npy".format(viewpointId, viewIndex, "_downscaled" if self.downscale_convolutional_features else ""))
                this_feats = np.load(path)
            else:
                # memmap for loading subfeatures
                path = os.path.join(cfs, scanId, "%s.npy" % viewpointId)
                mmapped = np.load(path, mmap_mode='r')
                this_feats = mmapped[viewIndex,:,:,:]
            feats.append(this_feats)
        if len(feats) > 1:
            return np.concatenate(feats, axis=1)
        return feats[0]

    def get_features(self, state):
        long_id = self._make_id(state.scanId, state.location.viewpointId)
        if self.image_feature_type == 'mean_pooled':
            return self.features[long_id][state.viewIndex,:]
        elif self.image_feature_type == 'attention':
            return self._get_convolutional_features(state.scanId, state.location.viewpointId, state.viewIndex)
        else:
            assert self.image_feature_type == 'none'
            return self.features

def read_visual_genome_vocab(fname, pad_name, add_null=False):
    # one-to-many mapping from indices to names (synonyms)
    index_to_items = []
    item_to_index = {}
    start_ix = 0
    items_to_add = [pad_name]
    if add_null:
        null_tp = ()
        items_to_add.append(null_tp)
    for item in items_to_add:
        index_to_items.append(item)
        item_to_index[item] = start_ix
        start_ix += 1

    with open(fname) as f:
        for index, line in enumerate(f):
            this_items = []
            for synonym in line.split(','):
                item = tuple(synonym.split())
                this_items.append(item)
                item_to_index[item] = index + start_ix
            index_to_items.append(this_items)
    assert len(index_to_items) == max(item_to_index.values()) + 1
    return index_to_items, item_to_index

class BottomUpImageFeatures(ImageFeatures):
    PAD_ITEM = ("<pad>",)
    feature_dim = 2048

    def __init__(self, number_of_detections, precomputed_cache_path=None):
        super(BottomUpImageFeatures, self).__init__("bottom_up", [])
        self.number_of_detections = number_of_detections
        self.index_to_attributes, self.attribute_to_index = read_visual_genome_vocab(paths.bottom_up_attribute_path, BottomUpImageFeatures.PAD_ITEM, add_null=True)
        self.index_to_objects, self.object_to_index = read_visual_genome_vocab(paths.bottom_up_object_path, BottomUpImageFeatures.PAD_ITEM, add_null=False)

        self.num_attributes = len(self.index_to_attributes)
        self.num_objects = len(self.index_to_objects)

        self.attribute_pad_index = self.attribute_to_index[BottomUpImageFeatures.PAD_ITEM]
        self.object_pad_index = self.object_to_index[BottomUpImageFeatures.PAD_ITEM]

        if precomputed_cache_path:
            self.precomputed_cache = {}
            with open(precomputed_cache_path, 'rb') as f:
                data = pickle.load(f)
                for (key, viewpoints) in data.items():
                    assert len(viewpoints) == ImageFeatures.NUM_VIEWS
                    viewpoint_feats = []
                    for viewpoint in viewpoints:
                        params = {}
                        for param_key, param_value in viewpoint.items():
                            assert len(param_value) >= self.number_of_detections
                            params[param_key] = param_value[:self.number_of_detections]
                        viewpoint_feats.append(BottomUpViewpoint(**params))
                    self.precomputed_cache[key] = viewpoint_feats
        else:
            self.precomputed_cache = None


    def batch_features(self, feature_list):
        def transform(lst, wrap_with_var=True):
            features = np.stack(lst)
            x = torch.from_numpy(features)
            if wrap_with_var:
                x = Variable(x, requires_grad=False)
            return try_cuda(x)

        return BottomUpViewpoint(
            cls_prob=transform([f.cls_prob for f in feature_list]),
            image_features=transform([f.image_features for f in feature_list]),
            attribute_indices=transform([f.attribute_indices for f in feature_list]),
            object_indices=transform([f.object_indices for f in feature_list]),
            boxes=transform([f.boxes for f in feature_list]),
            no_object_mask=transform([f.no_object_mask for f in feature_list], wrap_with_var=False),
        )

    def parse_attribute_objects(self, tokens):
        parse_options = []
        # allow blank attribute, but not blank object
        for split_point in range(0, len(tokens)):
            attr_tokens = tuple(tokens[:split_point])
            obj_tokens = tuple(tokens[split_point:])
            if attr_tokens in self.attribute_to_index and obj_tokens in self.object_to_index:
                parse_options.append((self.attribute_to_index[attr_tokens], self.object_to_index[obj_tokens]))
        assert parse_options, "didn't find any parses for {}".format(tokens)
        # prefer longer objects, e.g. "electrical outlet" over "electrical" "outlet"
        return parse_options[0]

    @functools.lru_cache(maxsize=20000)
    def _get_viewpoint_features(self, scan_id, viewpoint_id):
        if self.precomputed_cache is not None:
            return self.precomputed_cache[(scan_id, viewpoint_id)]

        fname = os.path.join(paths.bottom_up_feature_store_path, scan_id, "{}.p".format(viewpoint_id))
        with open(fname, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        viewpoint_features = []
        for viewpoint in data:
            top_indices = k_best_indices(viewpoint['cls_prob'], self.number_of_detections, sorted=True)[::-1]

            no_object = np.full(self.number_of_detections, True, dtype=np.uint8) # will become torch Byte tensor
            no_object[0:len(top_indices)] = False

            cls_prob = np.zeros(self.number_of_detections, dtype=np.float32)
            cls_prob[0:len(top_indices)] = viewpoint['cls_prob'][top_indices]
            assert cls_prob[0] == np.max(cls_prob)

            image_features = np.zeros((self.number_of_detections, ImageFeatures.MEAN_POOLED_DIM), dtype=np.float32)
            image_features[0:len(top_indices)] = viewpoint['features'][top_indices]

            boxes = np.zeros((self.number_of_detections, 4), dtype=np.float32)
            boxes[0:len(top_indices)] = viewpoint['boxes'][top_indices]

            object_indices = np.full(self.number_of_detections, self.object_pad_index)
            attribute_indices = np.full(self.number_of_detections, self.attribute_pad_index)

            for i, ix in enumerate(top_indices):
                attribute_ix, object_ix = self.parse_attribute_objects(list(viewpoint['captions'][ix].split()))
                object_indices[i] = object_ix
                attribute_indices[i] = attribute_ix

            viewpoint_features.append(BottomUpViewpoint(cls_prob, image_features, attribute_indices, object_indices, boxes, no_object))
        return viewpoint_features

    def get_features(self, state):
        viewpoint_features = self._get_viewpoint_features(state.scanId, state.location.viewpointId)
        return viewpoint_features[state.viewIndex]

class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, image_features, batch_size, beam_size):
        self.sims = []
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.image_features = image_features
        for i in range(batch_size):
            beam = []
            for j in range(beam_size):
                sim = make_sim(self.image_features.image_w, self.image_features.image_h, self.image_features.vfov)
                beam.append(sim)
            self.sims.append(beam)

    def sims_view(self, beamed):
        if beamed:
            return self.sims
        else:
            return (s[0] for s in self.sims)

    def newEpisodes(self, scanIds, viewpointIds, headings, beamed=False):
        assert len(scanIds) == len(viewpointIds)
        assert len(headings) == len(viewpointIds)
        assert len(scanIds) == len(self.sims)
        world_states = []
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            world_state = WorldState(scanId, viewpointId, heading, 0)
            if beamed:
                world_states.append([world_state])
            else:
                world_states.append(world_state)
            load_world_state(self.sims[i][0], world_state)
        assert len(world_states) == len(scanIds)
        return world_states

    def getStates(self, world_states, beamed=False):
        ''' Get list of states. '''
        def f(sim, world_state):
            load_world_state(sim, world_state)
            return sim.getState()
        return structured_map(f, self.sims_view(beamed), world_states, nested=beamed)

    def makeActions(self, world_states, actions, beamed=False):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        def f(sim, world_state, action):
            index, heading, elevation = action
            load_world_state(sim, world_state)
            sim.makeAction(index, heading, elevation)
            return get_world_state(sim)
        return structured_map(f, self.sims_view(beamed), world_states, actions, nested=beamed)

    def makeSimpleActions(self, simple_indices, beamed=False):
        ''' Take an action using a simple interface: 0-forward, 1-turn left, 2-turn right, 3-look up, 4-look down.
            All viewpoint changes are 30 degrees. Forward, look up and look down may not succeed - check state.
            WARNING - Very likely this simple interface restricts some edges in the graph. Parts of the
            environment may not longer be navigable. '''
        def f(sim, index):
            if index == 0:
                sim.makeAction(1, 0, 0)
            elif index == 1:
                sim.makeAction(0,-1, 0)
            elif index == 2:
                sim.makeAction(0, 1, 0)
            elif index == 3:
                sim.makeAction(0, 0, 1)
            elif index == 4:
                sim.makeAction(0, 0,-1)
            else:
                sys.exit("Invalid simple action %s" % index)
        structured_map(f, self.sims_view(beamed), simple_indices, nested=beamed)
        return None

class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, image_features, batch_size=100, seed=10, splits=['train'], tokenizer=None, beam_size=1):
        self.image_features = image_features
        self.data = []
        self.scans = []
        for item in load_datasets(splits):
            # Split multiple instructions into separate entries
            for j,instr in enumerate(item['instructions']):
                self.scans.append(item['scan'])
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instructions'] = instr
                if tokenizer:
                    self.tokenizer = tokenizer
                    new_item['instr_encoding'], new_item['instr_length'] = tokenizer.encode_sentence(instr)
                else:
                    self.tokenizer = None
                self.data.append(new_item)
        self.scans = set(self.scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)
        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()
        self.set_beam_size(beam_size)
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def set_beam_size(self, beam_size):
        # warning: this will invalidate the environment, self.reset() should be called afterward!
        try:
            invalid = (beam_size != self.beam_size)
        except:
            invalid = True
        if invalid:
            self.beam_size = beam_size
            self.env = EnvBatch(self.image_features, self.batch_size, beam_size)

    def _load_nav_graphs(self):
        ''' Load connectivity graph for each scan, useful for reasoning about shortest paths '''
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, sort_instr_length):
        batch = self.data[self.ix:self.ix+self.batch_size]
        if len(batch) < self.batch_size:
            random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        if sort_instr_length:
            batch = sorted(batch, key=lambda item: item['instr_length'], reverse=True)
        self.batch = batch

    def reset_epoch(self):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return (0, 0, 0) # do nothing
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        # Can we see the next viewpoint?
        for i,loc in enumerate(state.navigableLocations):
            if loc.viewpointId == nextViewpointId:
                # Look directly at the viewpoint before moving
                if loc.rel_heading > math.pi/6.0:
                      return (0, 1, 0) # Turn right
                elif loc.rel_heading < -math.pi/6.0:
                      return (0,-1, 0) # Turn left
                elif loc.rel_elevation > math.pi/6.0 and state.viewIndex//12 < 2:
                      return (0, 0, 1) # Look up
                elif loc.rel_elevation < -math.pi/6.0 and state.viewIndex//12 > 0:
                      return (0, 0,-1) # Look down
                else:
                      return (i, 0, 0) # Move
        # Can't see it - first neutralize camera elevation
        if state.viewIndex//12 == 0:
            return (0, 0, 1) # Look up
        elif state.viewIndex//12 == 2:
            return (0, 0,-1) # Look down
        # Otherwise decide which way to turn
        target_rel = self.graphs[state.scanId].node[nextViewpointId]['position'] - state.location.point
        target_heading = math.pi/2.0 - math.atan2(target_rel[1], target_rel[0]) # convert to rel to y axis
        if target_heading < 0:
            target_heading += 2.0*math.pi
        if state.heading > target_heading and state.heading - target_heading < math.pi:
            return (0,-1, 0) # Turn left
        if target_heading > state.heading and target_heading - state.heading > math.pi:
            return (0,-1, 0) # Turn left
        return (0, 1, 0) # Turn right

    def observe(self, world_states, beamed=False, include_teacher=True):
        #start_time = time.time()
        obs = []
        for i,states_beam in enumerate(self.env.getStates(world_states, beamed=beamed)):
            item = self.batch[i]
            obs_batch = []
            for state in states_beam if beamed else [states_beam]:
                assert item['scan'] == state.scanId
                feature = self.image_features.get_features(state)
                ob = {
                    'instr_id' : item['instr_id'],
                    'scan' : state.scanId,
                    'viewpoint' : state.location.viewpointId,
                    'viewIndex' : state.viewIndex,
                    'heading' : state.heading,
                    'elevation' : state.elevation,
                    'feature' : feature,
                    'step' : state.step,
                    'navigableLocations' : state.navigableLocations,
                    'instructions' : item['instructions'],
                }
                if include_teacher:
                    ob['teacher'] = self._shortest_path_action(state, item['path'][-1])
                if 'instr_encoding' in item:
                    ob['instr_encoding'] = item['instr_encoding']
                if 'instr_length' in item:
                    ob['instr_length'] = item['instr_length']
                obs_batch.append(ob)
            if beamed:
                obs.append(obs_batch)
            else:
                assert len(obs_batch) == 1
                obs.append(obs_batch[0])

        #end_time = time.time()
        #print("get obs in {} seconds".format(end_time - start_time))
        return obs

    def reset(self, sort=False, beamed=False, load_next_minibatch=True):
        ''' Load a new minibatch / episodes. '''
        if load_next_minibatch:
            self._next_minibatch(sort)
        assert len(self.batch) == self.batch_size
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        return self.env.newEpisodes(scanIds, viewpointIds, headings, beamed=beamed)

    def step(self, world_states, actions, beamed=False):
        ''' Take action (same interface as makeActions) '''
        return self.env.makeActions(world_states, actions, beamed=beamed)

    def shortest_paths_to_goals(self, starting_world_states, max_steps, expand_multistep_actions=True):
        world_states = starting_world_states
        obs = self.observe(world_states)

        all_obs = []
        all_actions = []
        for ob in obs:
            all_obs.append([ob])
            all_actions.append([])

        ended = np.array([False] * len(obs))
        for t in range(max_steps):
            actions = [ob['teacher'] for ob in obs]
            if expand_multistep_actions:
                actions = [FOLLOWER_ENV_ACTIONS[index_action_tuple(a)] for a in actions]
            world_states = self.step(world_states, actions)
            obs = self.observe(world_states)
            for i,ob in enumerate(obs):
                if not ended[i]:
                    all_obs[i].append(ob)
            for i,a in enumerate(actions):
                if not ended[i]:
                    all_actions[i].append(index_action_tuple(a))
                    if a == (0, 0, 0):
                        ended[i] = True
            if ended.all():
                break
        return all_obs, all_actions

    def gold_obs_actions_and_instructions(self, max_steps, load_next_minibatch=True, expand_multistep_actions=True):
        starting_world_states = self.reset(load_next_minibatch=load_next_minibatch)
        path_obs, path_actions = self.shortest_paths_to_goals(starting_world_states, max_steps, expand_multistep_actions=expand_multistep_actions)
        encoded_instructions = [obs[0]['instr_encoding'] for obs in path_obs]
        return path_obs, path_actions, encoded_instructions
