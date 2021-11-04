import json
from collections import defaultdict
from tasks.R2R.utils import read_vocab, Tokenizer
from tasks.R2R.vocab import TRAIN_VOCAB
import pickle
import re
import pprint


MAX_ELEMENTS_COUNT_IN_PATH = 3
MAX_OBJECTS_BY_SEGMENT = 8
FORBIDDEN_WORDS = [
  'doorframe', 'light', 'floor', 'ceiling', 'remove', 'otherroom',
  'roof', 'unknown', 'wall', 'door', 'rug', 'frame', 'column', 'window'
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
  sequence_length = elements[0]['sequence_len']
  for element in elements:
    # For the moment, we only process objects
    if 'area' not in element:
      continue

    by_segment_objects[element['sequence_index']].append({
      'name': element['name'],
      'area': element['area'],
      'distance': element['distance']
    })

  def segment_of_word(idx, instruction_length, sequence_length):
    return round(idx / instruction_length * sequence_length)

  by_word_objects = []
  instruction_length = len(instruction)
  for idx, word in enumerate(instruction):
    segment = segment_of_word(idx, instruction_length, sequence_length)
    objects = filter_objects(by_segment_objects[segment])

    by_word_objects.append(objects)

  return by_word_objects


processed_data = {}
for key, value in list(data.items())[:10]:

  instruction_objects_by_word = []
  for instruction_metadata in value['instructions_metadata']:
    instruction_objects_by_word.append(
      {
        'instruction': instruction_metadata['instruction'],
        'words_objects': get_word_objects(instruction_metadata)
      }
    )
  processed_data[key] = instruction_objects_by_word


import pdb; pdb.set_trace()


#with open('data/working_data/train_metadata.pickle', 'wb') as file:
#  pickle.dump(processed_data, file)

