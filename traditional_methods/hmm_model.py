#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from data_process import HMMDataLoader
import os


class HMM:
    def __init__(self, data_path="../seg-data/training/pku_training.utf8"):
        self.A = None   # 状态转移矩阵
        self.B = None   # 状态与观察序列转移矩阵
        self.Pi = None  # 初始状态概率
        self.data_loader = HMMDataLoader(data_path)
        self._get_index()

    def _get_index(self):
        self.idxed_corpus, (self.obsv2idx, self.idx2obsv), (self.hide2idx, self.idx2hide) = self.data_loader.index_corpus()
        self.num_obsv = len(self.obsv2idx.keys())
        self.num_hide = len(self.hide2idx.keys())
        print("Status dict:", self.hide2idx)

    def build_supervised_model(self, smooth="add1"):
        if smooth not in ['add1']:
            raise ValueError("Invalid value for smooth, only accept 'add1'.")

        if self.num_hide and self.num_obsv:
            self.Pi = np.zeros(self.num_hide)
            self.A = np.zeros([self.num_hide, self.num_hide])
            self.B = np.zeros([self.num_obsv, self.num_hide])
        else:
            self.Pi = None
            self.A = None
            self.B = None

        # 统计频率，计算A，B，Pi参数
        for seq in self.idxed_corpus:
            for i in range(len(seq)):
                obsv_cur, hide_cur = seq[i]

                if (i == 0):
                    self.Pi[hide_cur] += 1
                else:
                    obsv_pre, hide_pre = seq[i - 1]
                    self.A[hide_cur, hide_pre] += 1

                self.B[obsv_cur, hide_cur] += 1

        # Todo：增加其他平滑处理方案
        # +1平滑
        if smooth == 'add1':
            self.A += 1
            self.B += 1
            self.Pi += 1

            self.Pi /= self.Pi.sum()
            self.A /= self.A.sum(axis=1)[:, None]
            self.B /= self.B.sum(axis=1)[:, None]

        return self.A, self.B, self.Pi

    def get_status_seq(self, obsv_seq):
        return self._veterbi(obsv_seq)

    def _veterbi(self, obsv_seq):
        # 初始化
        len_seq = len(obsv_seq)
        f = np.zeros([len_seq, self.num_hide])
        f_arg = np.zeros([len_seq, self.num_hide], dtype=int)
        for i in range(0, self.num_hide):
            f[0, i] = self.Pi[i] * self.B[obsv_seq[0], i]
            f_arg[0, i] = 0
        # 动态规划求解
        for i in range(1, len_seq):
            for j in range(self.num_hide):
                fs = [f[i-1, k] * self.A[j, k] * self.B[obsv_seq[i], j] for k in range(self.num_hide)]
                f[i, j] = max(fs)
                f_arg[i, j] = np.argmax(fs)
        # 反向求解最好的隐藏序列
        hidden_seq = [0] * len_seq
        z = np.argmax(f[len_seq-1, self.num_hide-1])
        hidden_seq[len_seq-1] = z
        for i in reversed(range(1, len_seq)):
            z = f_arg[i, z]
            hidden_seq[i-1] = z
        return hidden_seq

    def cut_sentence(self, sentence):
        sentence = sentence.strip()
        idxed_seq = [self.obsv2idx[obsv] if obsv in self.obsv2idx.keys() else 0 for obsv in sentence]
        idxed_hide = self.get_status_seq(idxed_seq)
        hide = [self.idx2hide[idx] for idx in idxed_hide]
        assert len(sentence) == len(hide), "状态序列与观测序列长度不一致"

        words = []
        lo, hi = 0, 0
        for i in range(len(hide)):
            if hide[i] == 'B':
                lo = i
            elif hide[i] == 'E':
                hi = i + 1
                words.append(sentence[lo:hi])
            elif hide[i] == 'S':
                words.append(sentence[i:i + 1])

        if hide[-1] == 'B':
            words.append(sentence[-1])  # 处理 SB,EB
        elif hide[-1] == 'M':
            words.append(sentence[lo:-1])

        assert len(sentence) == len("".join(words)), "还原失败,长度不一致\n{0}\n{1}\n{2}".format(sentence, "".join(words),
                                                                                        "".join(hide))
        return words

    def save(self, path, name="pku"):
        """ 保存隐式马尔可夫模型 """
        if name == "pku":
            np.save(os.path.join(path, "Pi.npy"), self.Pi)
            np.save(os.path.join(path, "A.npy"), self.A)
            np.save(os.path.join(path, "B.npy"), self.B)
        elif name == "msr":
            np.save(os.path.join(path, "Pi_MSR.npy"), self.Pi)
            np.save(os.path.join(path, "A_MSR.npy"), self.A)
            np.save(os.path.join(path, "B_MSR.npy"), self.B)
        else:
            print("parameter 'name' must be pku or msr!")

    def load(self, path, name="pku"):
        """ 加载隐式马尔可夫模型 """
        if name == "pku":
            self.Pi = np.load(os.path.join(path, "Pi.npy"))
            self.A = np.load(os.path.join(path, "A.npy"))
            self.B = np.load(os.path.join(path, "B.npy"))
            self.num_obsv = self.B.shape[0]
            self.num_hide = self.B.shape[1]
        elif name == "msr":
            self.Pi = np.load(os.path.join(path, "Pi_MSR.npy"))
            self.A = np.load(os.path.join(path, "A_MSR.npy"))
            self.B = np.load(os.path.join(path, "B_MSR.npy"))
            self.num_obsv = self.B.shape[0]
            self.num_hide = self.B.shape[1]
        else:
            print("parameter 'name' must be pku or msr!")


# 训练并保存模型
# data_path="../seg-data/training/pku_training.utf8"
# data_path="../seg-data/training/msr_training.utf8"
# hmm = HMM(data_path)
# A, B, Pi = hmm.build_supervised_model()
# hmm.save("model", "msr")
#
# hmm.load("model", "msr")
# result = hmm.cut_sentence("共同创造美好的新世纪——二○○一年新年贺词")
# print(result)



