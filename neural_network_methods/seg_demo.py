from data_process import build_corpus
from utils import load_model, extend_maps
import os

checkpoint_path = "ckpts/"


class CorpusDict:
    def __init__(self, tag_name):
        self.tag_name = tag_name

        self.word2id = {}
        self.id2word = {}
        self.tag2id = {}
        self.id2tag = {}

        self.bilstm_word2id = {}
        self.bilstm_tag2id = {}
        self.bilstm_id2word = {}
        self.bilstm_id2tag = {}

        self.bilstmcrf_word2id = {}
        self.bilstmcrf_tag2id = {}
        self.bilstmcrf_id2word = {}
        self.bilstmcrf_id2tag = {}

        self.build_index()

    def build_index(self):
        print("读取数据...")
        _, _, self.word2id, self.tag2id = build_corpus(self.tag_name, "train")
        self.id2word = self.exchange_dict(self.word2id)
        self.id2tag = self.exchange_dict(self.tag2id)

        print("构造Bi-LSTM词典……")
        self.bilstm_word2id, self.bilstm_tag2id = extend_maps(self.word2id, self.tag2id, for_crf=False)
        self.bilstm_id2word = self.exchange_dict(self.bilstm_word2id)
        self.bilstm_id2tag = self.exchange_dict(self.bilstm_tag2id)

        print("构造Bi-LSTM+CRF词典……")
        self.bilstmcrf_word2id, self.bilstmcrf_tag2id = extend_maps(self.word2id, self.tag2id, for_crf=True)
        self.bilstmcrf_id2word = self.exchange_dict(self.bilstmcrf_word2id)
        self.bilstmcrf_id2tag = self.exchange_dict(self.bilstmcrf_tag2id)

    @staticmethod
    def exchange_dict(dic):
        exchange_dic = {v: k for k, v in dic.items()}
        return exchange_dic


class SegSentence:
    def __init__(self, query, n_layer, data_tag, model_name):
        self.query = [query]
        self.model_name = model_name

        self.corpus_dict = CorpusDict(data_tag)
        self.model_path = self.get_model_path(n_layer, data_tag, model_name)
        self.model = load_model(self.model_path)
        self.pred = self.get_model_pred()

    def get_model_pred(self):
        query = self.query[0]
        query = list(query)
        query = [query]
        print(query)
        if self.model_name == "crf":
            pred = self.model.test(self.query)
        elif self.model_name == "bilstm":
            self.model.model.bilstm.flatten_parameters()  # remove warning
            pred = self.model.get_model_pred(query,
                                             self.corpus_dict.bilstm_word2id,
                                             self.corpus_dict.bilstm_tag2id)
        else:
            self.model.model.bilstm.bilstm.flatten_parameters()  # remove warning
            query[0].append('<end>')
            pred = self.model.get_model_pred(query,
                                             self.corpus_dict.bilstmcrf_word2id,
                                             self.corpus_dict.bilstmcrf_tag2id)
        return pred

    def get_model_path(self, n_layer, tag_name, model_name):
        assert n_layer in [1, 2, 3]
        assert tag_name in ["pku", "msr"]
        assert model_name in ["crf", "bilstm", "bilstm_crf"]

        model_dir_path = checkpoint_path + "param_128_128_" + str(n_layer) + "_" + tag_name
        model_path = os.path.join(model_dir_path, model_name + '.pkl')
        return model_path

    def cut(self):
        sent = self.query[0]
        tag = self.pred[0]
        words = []
        lo, hi = 0, 0
        for i in range(len(tag)):
            if tag[i] == 'B':
                lo = i
            elif tag[i] == 'E':
                hi = i + 1
                words.append(sent[lo:hi])
            elif tag[i] == 'S':
                words.append(sent[i:i + 1])

        if tag[-1] == 'B':
            words.append(sent[-1])  # 处理 SB,EB
        elif tag[-1] == 'M':
            words.append(sent[lo:-1])

        assert len(sent) == len("".join(words)), "还原失败,长度不一致\n{0}\n{1}\n{2}".format(sent, "".join(words),
                                                                                    "".join(tag))
        res = "/".join(words)
        return res


if __name__ == "__main__":
    query = "他一把把把把住了"
    n_layer = 1
    tag_name = "pku"
    # model_name = "crf"
    model_name = "bilstm"
    seg = SegSentence(query, n_layer, tag_name, model_name)
    # print(seg.pred)

    result = seg.cut()
    print(result)
