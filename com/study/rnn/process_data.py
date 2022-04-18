import re
import numpy as np
import pandas as pd
import jieba
from torch.utils.data import TensorDataset, DataLoader
import torch

batch_size = 32
word2idx = pd.read_pickle("/Users/ljk/Documents/code/pytorch/pytorch-study/com/study/rnn/word2idx.ckpt")
idx2word = pd.read_pickle("/Users/ljk/Documents/code/pytorch/pytorch-study/com/study/rnn/idx2word.ckpt")

# 读取数据
train_file = pd.read_csv("/Users/ljk/Documents/code/pytorch/pytorch-study/com/study/rnn/data/ChnSentiCorp/train.tsv",
                         delimiter="\t")
test_file = pd.read_csv("/Users/ljk/Documents/code/pytorch/pytorch-study/com/study/rnn/data/ChnSentiCorp/dev.tsv",
                        delimiter="\t")

# 将数据全部转化为小写
train_sentences = [str(x).lower() for x in train_file["text_a"]]
test_sentences = [str(x).lower() for x in test_file["text_a"]]

train_labels = [x for x in train_file["label"]]
test_labels = [x for x in test_file["label"]]
# 去除掉没有用的信息（数字、网址）
for i in range(len(train_sentences)):
    # 去除数字
    train_sentences[i] = re.sub('\d', '0', train_sentences[i])
    # 去除网址
    if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in \
            train_sentences[i]:
        train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])

for i in range(len(test_sentences)):
    test_sentences[i] = re.sub('\d', '0', test_sentences[i])
    if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in \
            test_sentences[i]:
        test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])

# 单词转化为数字
for i, sentence in enumerate(train_sentences):
    train_sentences[i] = [word2idx[word] if word in word2idx else 0 for word in jieba.lcut(sentence)]

for i, sentence in enumerate(test_sentences):
    test_sentences[i] = [word2idx[word.lower()] if word.lower() in word2idx else 0 for word in
                         jieba.lcut(sentence)]


# 固定所有句子的长度，这里选择200作为句子的固定长度，对于长度不够的句子，在前面填充0(_PAD)，超出长度的句子进行从后面截断
def pad_input(sentences, seq_len):
    features = np.ones((len(sentences), seq_len), dtype=int) * 148372
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, 0:len(review)] = np.array(review)[:seq_len]
    return features


# 固定测试数据集和训练数据集的句子长度
train_sentences = pad_input(train_sentences, 200)
test_sentences = pad_input(test_sentences, 200)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

print("-----------------训练样本数量：%d" % (len(train_sentences)))
print("-----------------训练样本数量：%d" % (len(train_labels)))
print("-----------------测试样本数量：%d" % (len(test_sentences)))
print("-----------------测试样本数量：%d" % (len(test_labels)))

# print(train_sentences[0:2])

train_data = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(train_labels))
test_data = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(test_labels))

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
