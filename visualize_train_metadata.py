import pickle
from tasks.R2R.utils import read_vocab, Tokenizer
from tasks.R2R.vocab import TRAIN_VOCAB

vocab = read_vocab(TRAIN_VOCAB)
tok = Tokenizer(vocab=vocab)

with open('data/working_data/train_objects_by_word.pickle', 'rb') as file:
  data = pickle.load(file)

print(len(data))
print(sum([len(data[key]) for key in data]))

print('-----')

with open('data/working_data/train_objects_by_word_objects_only.pickle', 'rb') as file:
  data = pickle.load(file)

print(len(data))
print(sum([len(data[key]) for key in data]))
