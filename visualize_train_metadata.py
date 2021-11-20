import pickle
from tasks.R2R.utils import read_vocab, Tokenizer
from tasks.R2R.vocab import TRAIN_VOCAB

vocab = read_vocab(TRAIN_VOCAB)
tok = Tokenizer(vocab=vocab)

with open('data/working_data/train_metadata.pickle', 'rb') as file:
  data = pickle.load(file)

for key in data.keys():
  for elements in data[key]['instructions_metadata']:
    for element in elements['elements']:
      if 'area' not in element:
        print(key)

