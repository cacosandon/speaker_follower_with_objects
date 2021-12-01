import pickle
from tasks.R2R.utils import read_vocab, Tokenizer
from tasks.R2R.vocab import TRAIN_VOCAB

vocab = read_vocab(TRAIN_VOCAB)
tok = Tokenizer(vocab=vocab)
