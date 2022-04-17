#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from hmm_model import HMM

from data_process import HMMDataLoader
from unigram_model import UniGramSeg
import os
import timeit


def article_hmm_seg(test_path, save_name, name, data_path, encoding="utf-8"):
    hmm = HMM(data_path=data_path)
    hmm.load("model", name)
    cut = hmm.cut_sentence
    pred = []
    ch_count = 0
    with open(test_path, "r", encoding=encoding) as f:
        lines = f.readlines()
        start = timeit.default_timer()
        count = 0
        total_count = len(lines)
        for sent in lines:
            try:
                pred.append(cut(sent))
                ch_count += len(sent)
            except:
                pred.append(["Error"] * len(sent))
                continue
            count += 1
            if count % 500 == 0:
                print("Processed {0}/{1} ---- {2}".format(count, total_count, count / total_count))
        end = timeit.default_timer()
        time_cost = end - start

    print("Total number of characters: {0}.".format(ch_count))
    print("Time cost: {0}s.".format(time_cost))
    print("Processed characters per second: {0}.".format(int(ch_count / time_cost)))

    # 保存分词结果
    save_path = os.path.join('test_result', save_name + '_seg.txt')
    # with open(save_path, "w", encoding='gb18030') as f:
    with open(save_path, "w", encoding='utf-8') as f:
        for words in pred:
            s = " ".join(words)
            s = s.strip() + '\n'
            # s.encode('gb18030')
            s.encode('utf-8')
            f.write(s)

    print("Segmentation result is saved in {0}.".format(save_path))


def article_unigram_seg(test_path, save_name, data_path="../seg-data/training/pku_training.utf8", encoding="utf-8"):
    seg = UniGramSeg()
    data_loader = HMMDataLoader(data_path)
    vocab_dict = data_loader.generate_vocab_dict()
    seg.set_dict(vocab_dict)

    pred = []
    ch_count = 0
    with open(test_path, "r", encoding=encoding) as f:
        lines = f.readlines()
        start = timeit.default_timer()
        count = 0
        total_count = len(lines)
        for sent in lines:
            try:
                pred.append(seg.cut(sent))
                ch_count += len(sent)
            except:
                pred.append(["Error"] * len(sent))
                continue
            count += 1
            if count % 500 == 0:
                print("Processed {0}/{1} ---- {2}".format(count, total_count, count / total_count))
        end = timeit.default_timer()
        time_cost = end - start

    print("Total number of characters: {0}.".format(ch_count))
    print("Time cost: {0}s.".format(time_cost))
    print("Processed characters per second: {0}.".format(int(ch_count / time_cost)))

    # 保存分词结果
    save_path = os.path.join('test_result', save_name + '_seg.txt')
    # with open(save_path, "w", encoding='gb18030') as f:
    with open(save_path, "w", encoding='utf-8') as f:
        for words in pred:
            s = " ".join(words)
            s = s.strip() + '\n'
            # s.encode('gb18030')
            s.encode('utf-8')
            f.write(s)

    print("Segmentation result is saved in {0}.".format(save_path))


if __name__ == "__main__":
    # test_path = "../seg-data/testing/pku_test.utf8"
    test_path = "../seg-data/testing/msr_test.utf8"
    # data_path = "../seg-data/training/pku_training.utf8"
    data_path = "../seg-data/training/msr_training.utf8"
    # name = "pku"
    name = "msr"

    # save_name = "hmm_pku"
    # save_name = "unigram_pku"
    # save_name = "unigram_raw_pku"
    # save_name = "hmm_msr"
    save_name = "unigram_msr"
    # save_name = "unigram_raw_msr"
    # article_hmm_seg(test_path, save_name, name, data_path)
    article_unigram_seg(test_path, save_name, data_path)


