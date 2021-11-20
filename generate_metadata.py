import json
from collections import defaultdict
from tasks.R2R.utils import read_vocab, Tokenizer
from tasks.R2R.vocab import TRAIN_VOCAB
import pickle
import re
import pprint


MAX_ELEMENTS_COUNT_IN_PATH = 1
MAX_OBJECTS_BY_SEGMENT = 8
FORBIDDEN_WORDS = [
  'doorframe', 'light', 'floor', 'ceiling', 'remove', 'otherroom',
  'roof', 'unknown', 'wall', 'door', 'rug', 'frame', 'column', 'window', 'column', 'celing'
]

with open('data/working_data/train_metadata.pickle', 'rb') as file:
  data = pickle.load(file)

"""
{
  path_id: [
    {
      instruction: 'blabla',
      elements: [...]
    }
  ]
}
"""

vocab = read_vocab(TRAIN_VOCAB)
tok = Tokenizer(vocab=vocab)

def filter_elements_by_count(elements):
  elements_names = list(map(lambda x: x['name'], elements))

  def obj_count_more_than_limit(obj):
    return elements_names.count(obj['name']) <= MAX_ELEMENTS_COUNT_IN_PATH

  return list(filter(obj_count_more_than_limit, elements))


def filter_objects(objects):
  final_objects = []

  def get_area(object):
    return object['area']

  def check_permitted(object):
    first_boolean = all(forbidden not in object['name'] for forbidden in FORBIDDEN_WORDS)
    second_boolean = 1 not in tok.encode_sentence(object['name'])[0]
    third_boolean = object['distance'] < 5
    return all([first_boolean, second_boolean, third_boolean])

  objects = list(filter(check_permitted, objects))
  sorted_objects_by_area = objects.sort(key = get_area, reverse = True)

  final_objects = objects[:MAX_OBJECTS_BY_SEGMENT]
  return final_objects

def get_word_objects(instruction_metadata):
  instruction, instr_len = tok.encode_sentence(instruction_metadata['instruction'])
  elements = instruction_metadata['elements']

  elements = filter_elements_by_count(elements)

  by_segment_objects = defaultdict(list)

  if elements:
    sequence_length = elements[0]['sequence_len']
  else:
    sequence_length = instr_len
  for element in elements:

    # Only objects
    if 'area' not in element:
      continue

    by_segment_objects[element['sequence_index']].append({
      'name': element['name'],
      'area': element['area'],
      'distance': element['distance']
    })

  def segment_of_word(idx, instruction_length, sequence_length):
    return round(idx / instr_len * sequence_length)

  by_word_objects = []
  instr_len = len(instruction)
  for idx, word in enumerate(instruction):
    segment = segment_of_word(idx, instr_len, sequence_length)
    objects = filter_objects(by_segment_objects[segment])

    by_word_objects.append(objects)

  return by_word_objects

def get_word_viewpoints(instruction_metadata):
  print(instruction_metadata)
  instruction, instr_len = tok.encode_sentence(instruction_metadata['instruction'])
  elements = instruction_metadata['elements']

  elements = filter_elements_by_count(elements)

  by_segment_viewpoints = defaultdict(list)

  if elements:
    sequence_length = elements[0]['sequence_len']
  else:
    sequence_length = instr_len
  for element in elements:
    # Only viewpoints
    if 'area' in element:
      continue

    by_segment_viewpoints[element['sequence_index']].append({
      'name': element['name'],
      'distance': element['distance']
    })

  def segment_of_word(idx, instruction_length, sequence_length):
    return round(idx / instr_len * sequence_length)

  by_word_viewpoints = []
  instr_len = len(instruction)
  for idx, word in enumerate(instruction):
    segment = segment_of_word(idx, instr_len, sequence_length)
    # objects = filter_viewpoints(by_segment_objects[segment])

    by_word_viewpoints.append(by_segment_viewpoints[segment])

  return by_word_viewpoints

processed_data = {}
for key, value in data.items():

  if key != 6597:
    continue

  instruction_objects_by_word = []
  for instruction_metadata in value['instructions_metadata']:
    instruction_objects_by_word.append(
      {
        'instruction': instruction_metadata['instruction'],
        'words_objects': get_word_objects(instruction_metadata),
        'words_viewpoints': get_word_viewpoints(instruction_metadata)
      }
    )
  processed_data[key] = instruction_objects_by_word
  break

print(processed_data)


#with open('data/working_data/train_objects_by_word.pickle', 'wb') as file:
#  pickle.dump(processed_data, file)

