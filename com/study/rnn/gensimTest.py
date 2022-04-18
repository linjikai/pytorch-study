import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import pandas as pd

w2v_model = KeyedVectors.load_word2vec_format(
    "/Users/ljk/Documents/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.bz2",
    binary=False,
    unicode_errors="Ignore")

word_embed_save_path = "/Users/ljk/Documents/code/pytorch/pytorch-study/com/study/rnn/bd.dim300.ckpt.npy"
np.save(word_embed_save_path,w2v_model.vectors)

pd.to_pickle(w2v_model.index_to_key,"/Users/ljk/Documents/code/pytorch/pytorch-study/com/study/rnn/idx2word.ckpt")
pd.to_pickle(w2v_model.key_to_index,"/Users/ljk/Documents/code/pytorch/pytorch-study/com/study/rnn/word2idx.ckpt")


# print(len(w2v_model.key_to_index.keys())) #635963
# print(w2v_model.get_vector(","))
# print(w2v_model.get_vector("中国"))
# print(w2v_model.get_vector("中"))
# print(w2v_model.get_vector("国"))
# print(w2v_model.get_vector("a"))
# print(w2v_model.get_vector("b"))
