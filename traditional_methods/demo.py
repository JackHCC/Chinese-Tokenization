#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from hmm_model import HMM
from data_process import HMMDataLoader
from unigram_model import UniGramSeg

data_path="../seg-data/training/pku_training.utf8"
hmm = HMM(data_path)
hmm.load("model")
cut = hmm.cut_sentence

seg = UniGramSeg()
data_loader = HMMDataLoader(data_path)
vocab_dict = data_loader.generate_vocab_dict()
seg.set_dict(vocab_dict)

s = ""
print("这是一个分词Demo, 输入'q'退出.")
while s != 'q':
    s = input("请输入待分词句子: ")
    q = input("选择模型 hmm or ngram?")
    if q == "hmm":
        print("HMM分词结果：")
        words = cut(s)
    elif q == "ngram":
        print("N-gram分词结果：")
        words = seg.cut(s)
    else:
        print("模式输入错误！")
        continue
    print("/".join(words))
