import numpy as np
import torch
from torch import nn
import pandas as pd

weight_numpy = np.load("/Users/ljk/Documents/code/pytorch/pytorch-study/com/study/rnn/bd.dim300.ckpt.npy")
embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weight_numpy))

word2idx = pd.read_pickle("/Users/ljk/Documents/code/pytorch/pytorch-study/com/study/rnn/word2idx.ckpt")
idx2word = pd.read_pickle("/Users/ljk/Documents/code/pytorch/pytorch-study/com/study/rnn/idx2word.ckpt")

sentences = ["我", "爱", "北京", "天安门"]

ids = torch.LongTensor([word2idx[item] for item in sentences])

word_vec = embedding(ids)

print(word_vec.shape)
print(type(word_vec))

print(word2idx["PAD"])
