#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import json
from collections import Counter
import matplotlib.pyplot as plt


def json_read(file_path, encoding="uft-8"):
    """ 读取json文件解析为dict返回.
    """
    with open(file_path, 'r', encoding=encoding) as f:
        return json.load(f)


def dict_save(mdict, save_path="dict.json"):
    """ 将词典保存至json文件中

        Args:
          mdict: 待保存词典
          save_path: 词典将保存在该路径下
    """
    with open(save_path, 'w', encoding="utf-8") as f:
        json.dump(mdict, f)
        print("Dict is saved in -- {0}.".format(save_path))


class HMMDataLoader:
    def __init__(self, data_path):
        # self.status2num = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
        self.data_path = data_path
        self.vocab_dict = {}
        self.corpus = []

    def generate_vocab_dict(self, encoding='utf-8'):
        """
        统计语料库，获取词表
        :param encoding: 编码方式
        :return: 词表字典
        """
        with open(self.data_path, 'r', encoding=encoding) as f:
            print("Start generate vocab dict...")

            for line in f.readlines():
                for word in line.strip().split():
                    self.vocab_dict[word] = self.vocab_dict.get(word, 0) + 1
            count = len(self.vocab_dict)
            self.vocab_dict['_total_'] = count
            print("Finished. Total number of words: {0}".format(count))

        return self.vocab_dict

    def generate_corpus_status(self, encoding="utf-8"):
        """ 将原始分词数据集处理成BMES标记的数据集.

            返回格式[[('你','B'),..],[..],..]
        """
        with open(self.data_path, 'r', encoding=encoding) as f:
            for line in f:
                l = []
                for word in line.strip().split():
                    if len(word) == 1:
                        l.append((word[0], 'S'))
                        continue
                    for i in range(len(word)):
                        if i == 0:
                            l.append((word[i], 'B'))
                        elif i == len(word) - 1:
                            l.append((word[i], 'E'))
                        else:
                            l.append((word[i], 'M'))
                self.corpus.append(l)
        return self.corpus

    def index_corpus(self):
        """ 将数据集进行编码

            Args:
              corpus: 格式必须为[[(obsv, hide), (obsv,hide),...], ...]

            Returns:
              idxed_corpus: 编码后的corpus,格式不变
              (obsv2idx, idx2obsv): 两个dict用于观察值与其编码之间的转换
              (hide2idx, idx2hide): 两个dict用于隐藏值与其编码之间的转换
        """
        obsv2idx, idx2obsv = {'unk': 0}, {0: 'unk'}
        hide2idx, idx2hide = {}, {}
        obsv_idx, hide_idx = 1, 0

        # build dictionaries and indexing
        idxed_corpus = []
        for seq in self.generate_corpus_status():
            idxed_seq = []
            for obsv, hide in seq:
                if obsv not in obsv2idx.keys():
                    obsv2idx[obsv] = obsv_idx
                    idx2obsv[obsv_idx] = obsv
                    obsv_idx += 1
                if hide not in hide2idx.keys():
                    hide2idx[hide] = hide_idx
                    idx2hide[hide_idx] = hide
                    hide_idx += 1
                # indexing
                idxed_seq.append((obsv2idx[obsv], hide2idx[hide]))
            idxed_corpus.append(idxed_seq)

        return idxed_corpus, (obsv2idx, idx2obsv), (hide2idx, idx2hide)

    def draw_zipf(self):
        vocab_dict = self.generate_vocab_dict()
        del vocab_dict["_total_"]
        counter = Counter(vocab_dict.values())
        counter_top = counter.most_common()
        plt.figure('Word frequent')
        plt.rc('font', family='SimHei', size=13)
        y = list(map(lambda y: y[1], counter_top[:]))
        plt.xlabel('rank')
        plt.ylabel('frequency')
        plt.plot(range(len(y)), y)
        plt.scatter(range(len(y)), y, s=2, c='red')
        plt.title('Word frequency vs rank')
        plt.show()

    def draw_top10_hist(self):
        vocab_dict = self.generate_vocab_dict()
        del vocab_dict["_total_"]
        counter = Counter(vocab_dict)
        counter_top = counter.most_common(10)
        x = list(map(lambda x: x[1], counter_top[:]))
        y = list(map(lambda y: y[0], counter_top[:]))
        plt.rc('font', family='SimHei', size=13)
        plt.barh(range(10), x, height=0.7, color='steelblue', alpha=0.8)  # 从下往上画
        plt.yticks(range(10), y)
        plt.xlabel("频数")
        plt.title("频率前10的词语")
        for x, y in enumerate(x):
            plt.text(y + 0.2, x - 0.1, '%s' % y)
        plt.show()



# 统计语料词表
# data_path = "../seg-data/training/pku_training.utf8"
# data_path = "../seg-data/training/msr_training.utf8"
# save_path = "./dict/pku_vocab_dict.json"
# save_path = "./dict/msr_vocab_dict.json"
# hmm_dataloader = HMMDataLoader(data_path)
# dict_save(hmm_dataloader.generate_vocab_dict(), save_path)
# idxed_corpus, (obsv2idx, idx2obsv), (hide2idx, idx2hide) = hmm_dataloader.index_corpus()
# print(len(idxed_corpus))
# print(len(obsv2idx))
# print(len(hide2idx))
# print(idxed_corpus[:10])

# vocab_dict = json_read(save_path)

# 数据集统计分析图
# hmm_dataloader.draw_zipf()
# hmm_dataloader.draw_top10_hist()



