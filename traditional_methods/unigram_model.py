#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import re
import math
from data_process import HMMDataLoader


class UniGramSeg:
    """ 基于词典以及1-gram的中文分词器. """

    def __init__(self):
        self.dict = None
        self.punc = ['———', '、', '，', '。', '！', '：']

    def set_dict(self, mdict):
        """ 设置词典. """
        self.dict = mdict

    def cut(self, sent, smooth='add1'):
        """ 切分给定字符串,返回切分结果.

            Args:
              sent: 待切分字符串.
              smooth: 选择不同的平滑算法,可选值有:.

            Returns:
              切分结果,一个列表,包含切分出来的各个词语.
        """
        if smooth not in ['add1']:
            raise ValueError("invalid value for smooth, only accept 'add1'")

        if smooth == 'add1':
            words = []
            sents = self.__shorten_sent(sent.strip())
            for s in sents:
                if s in self.punc:
                    words += s
                elif s == '':
                    continue
                else:
                    words.extend(self.__cut(s))
            return words
        else:
            raise NotImplementedError()

    def __shorten_sent(self, sent):
        pattern = r'(' + "|".join(self.punc) + r')'
        return re.split(pattern, sent)

    def __cut(self, sent):

        sent = sent.strip()

        log = lambda x: float('-inf') if not x else math.log(x)
        # freq = lambda x: self.dict[x] if x in self.dict else 0 if len(x)>1 else 1  # 计算每个词的频次（加入平滑）

        l = len(sent)
        maxsum = [0] * (l + 1)  # 以i-1为一个词的结尾的最大log和
        cp = [0] * (l + 1)  # cut point(分割点的位置)

        # DP
        for i in range(1, l + 1):
            # 不加规则
            # maxsum[i], cp[i] = max([(log(self.freq_raw(sent[k:i]) / self.dict['_total_']) + maxsum[k], k) for k in range(0, i)])
            # 加规则
            maxsum[i], cp[i] = max([(log(self.freq(sent[k:i]) / self.dict['_total_']) + maxsum[k], k) for k in range(0, i)])

        # 回溯构建分出来的词语
        words = []
        lo, hi = 0, l
        while hi != 0:
            lo = cp[hi]
            words.append(sent[lo:hi])
            hi = lo

        return list(reversed(words))

    def __is_date_or_number(self, string):
        '''
        判断一个字符串是否为数字或者日期
        '''
        length = len(string)
        if length < 2:
            return 0

        if string.isdigit():
            return 1
        elif ((string[length - 1] == '年' or string[length - 1] == '月' or string[length - 1] == '日' or string[
            length - 1] == '时'
               or string[length - 1] == '分') and string[0:length - 2].isdigit()):
            return 1
        elif (length == 2 and (string[length - 1] == '年' or string[length - 1] == '月' or string[length - 1] == '日'
                               or string[length - 1] == '时' or string[length - 1] == '分') and string[0].isdigit()):
            return 1
        else:
            return 0

    def freq(self, x):
        # print(x)
        if x in self.dict:
            return self.dict[x]
        elif self.__is_date_or_number(x) or self.__special_string_handle(x):
            return max(self.dict.values()) / len(self.dict)
        elif len(x) > 1:
            return 0  # 不是单字的没有概率
        else:
            return 1  # 单字的不在词典的话有1

    def freq_raw(self, x):
        # print(x)
        if x in self.dict:
            return self.dict[x]
        elif len(x) > 1:
            return 0  # 不是单字的没有概率
        else:
            return 1  # 单字的不在词典的话有1

    def __judge_string(self, check_str):
        for ch in check_str:
            if 'a' <= ch <= 'z' or 'A' <= ch <= 'Z' or '0' <= ch <= '9' or ch == '％' or ch == '．' or ch == '@' or ch == '.':
                continue
            else:
                return 0
                break
        return 1

    def __special_string_handle(self, string):
        '''
        判断一个字符串是否为特殊情况
        '''
        list1 = ['月', '日', '时', '分']
        list2 = ['亿', '万']
        length = len(string)
        if length < 2:
            return 0
        # print(length)

        # 不包含中文和℃的情况
        if self.__judge_string(string) == 1:
            return 1
        # 日期的情况 其中得派出并不是指时间的年 例如“过去了90年” 长度的判断 需要分开写
        elif string[length - 1] == '年' and length >= 5 and string[0:length - 2].isdigit():
            return 1
        elif string[length - 1] in list1 and string[0:length - 2].isdigit():
            return 1
        elif length == 2 and string[length - 1] in list1 and string[0].isdigit():
            return 1
        # 数量单位的处理“亿” “万” “万亿”
        elif string[length - 1] in list2 and self.__judge_string(string[0:length - 2]) == 1:
            return 1
        elif length >= 3 and string[length - 2:length - 3] == '万亿' and self.__judge(string[0:length - 2]) == 1:
            return 1
        else:
            return 0

s = "共同创造美好的新世纪——二○○一年新年贺词"
seg = UniGramSeg()
data_path="../seg-data/training/pku_training.utf8"
data_loader = HMMDataLoader(data_path)
vocab_dict = data_loader.generate_vocab_dict()
seg.set_dict(vocab_dict)

words = seg.cut(s)
# print(words)
print("/".join(words))
