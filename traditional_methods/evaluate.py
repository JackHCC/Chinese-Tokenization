#!/usr/bin/env python
# -*- coding: UTF-8 -*-

def precision(pred, std):

    correct = 0
    total_num = 0

    for p_s, std_s in zip(pred, std):
        for word in p_s:
            if word in std_s:
                correct += 1
        total_num += len(p_s)

    return float(correct) / total_num


def recall(pred, std):

    correct = 0
    total_num = 0

    for p_s, std_s in zip(pred, std):
        for word in p_s:
            if word in std_s:
                correct += 1
        total_num += len(std_s)

    return float(correct) / total_num


def f1_score(pred, std):

    prec = precision(pred, std)
    rec = recall(pred, std)

    return 2.0 * prec * rec / (prec + rec)


def evaluate(test_path, gold_path, encoding="utf-8"):
    pred = []
    with open(test_path, "r", encoding=encoding) as f:
        lines = f.readlines()
        for sent in lines:
            pred.append(sent.strip().split())

    std = []
    with open(gold_path, "r", encoding=encoding) as f:
        for sent in f.readlines():
            std.append(sent.strip().split())

    print("Precision: {0}.".format(precision(pred, std)))
    print("Recall: {0}.".format(recall(pred, std)))
    print("F1 Score: {0}.".format(f1_score(pred, std)))


if __name__ == "__main__":
    # # test_path = "./test_result/hmm_pku_seg.txt"
    # # test_path = "./test_result/unigram_pku_seg.txt"
    # test_path = "./test_result/unigram_raw_pku_seg.txt"

    # test_path = "./test_result/hmm_msr_seg.txt"
    test_path = "./test_result/unigram_msr_seg.txt"
    # test_path = "./test_result/unigram_raw_msr_seg.txt"

    # gold_path = "../seg-data/gold/pku_test_gold.utf8"
    gold_path = "../seg-data/gold/msr_test_gold.utf8"
    evaluate(test_path, gold_path)

