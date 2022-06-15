import os, shutil
from data_process import build_corpus
from utils import extend_maps, prepocess_data_for_lstmcrf
from evaluate import crf_train_eval, bilstm_train_and_eval, ensemble_evaluate


def main():
    """训练模型，评估结果"""
    output_name = "param_128_128_1_pku"
    data_tag = "pku"
    output_directory = os.path.join('ckpts', output_name)

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    shutil.copy2('models/config.py', output_directory)

    # 读取数据
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus(data_tag, "train", fix_length=-1)
    dev_word_lists, dev_tag_lists = build_corpus(data_tag, "dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus(data_tag, "test", make_vocab=False)


    # 训练评估CRF模型
    print("正在训练评估CRF模型...")
    crf_pred = crf_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists),
        output_directory
    )

    # 训练评估BI-LSTM模型
    print("正在训练评估双向LSTM模型...")
    # LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    lstm_pred = bilstm_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        bilstm_word2id, bilstm_tag2id,
        output_directory,
        crf=False
    )

    print("正在训练评估Bi-LSTM+CRF模型...")
    # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    # 还需要额外的一些数据处理
    train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(
        train_word_lists, train_tag_lists
    )
    dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(
        dev_word_lists, dev_tag_lists
    )
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True
    )
    lstmcrf_pred = bilstm_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        crf_word2id, crf_tag2id,
        output_directory
    )

    ensemble_evaluate(
        [crf_pred, lstm_pred, lstmcrf_pred],
        test_tag_lists
    )


if __name__ == "__main__":
    main()
