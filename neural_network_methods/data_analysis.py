import matplotlib.pyplot as plt


class DataStatics:
    def __init__(self, path, encoding="utf-8"):
        self.data_path = path
        self.encoding = encoding
        self.vocab_dict = {}
        self.count = 0
        self.punctuation = ['，', '。', '：', '；', '？', '！', '、', '“', '”', '……', '（', '）']
        self.uni_gram_map = {}
        self.bi_gram_map = {}
        self.tri_gram_map = {}

        self.build_vocab_dict()
        self.statics()
        self.draw_hist()

    def build_vocab_dict(self):
        with open(self.data_path, 'r', encoding=self.encoding) as f:
            print("Start generate vocab dict...")
            for line in f.readlines():
                for word in line.strip().split():
                    self.vocab_dict[word] = self.vocab_dict.get(word, 0) + 1
            self.count = len(self.vocab_dict)
            print("Finished. Total number of words: {0}".format(self.count))

    def statics(self):
        uni_gram_dict = {}
        bi_gram_dict = {}
        tri_gram_dict = {}
        for key, value in self.vocab_dict.items():
            if len(key) == 1 and key not in self.punctuation:
                uni_gram_dict[key] = value
            elif len(key) == 2:
                bi_gram_dict[key] = value
            elif len(key) == 3:
                tri_gram_dict[key] = value

        self.uni_gram_map = {k: v for k, v in sorted(uni_gram_dict.items(), key=lambda item: item[1], reverse=True)}
        self.bi_gram_map = {k: v for k, v in sorted(bi_gram_dict.items(), key=lambda item: item[1], reverse=True)}
        self.tri_gram_map = {k: v for k, v in sorted(tri_gram_dict.items(), key=lambda item: item[1], reverse=True)}

    def draw_hist(self):
        # 一元词
        plt.figure(1)
        x = list(map(lambda x: x[1], list(self.uni_gram_map.items())[:10]))
        y = list(map(lambda y: y[0], list(self.uni_gram_map.items())[:10]))
        plt.rc('font', family='SimHei', size=13)
        plt.barh(range(10), x, height=0.7, color='steelblue', alpha=0.8)  # 从下往上画
        plt.yticks(range(10), y)
        plt.xlabel("频数")
        plt.title("频率前10的词语")
        for x, y in enumerate(x):
            plt.text(y + 0.2, x - 0.1, '%s' % y)

        # 二元词
        plt.figure(2)
        x = list(map(lambda x: x[1], list(self.bi_gram_map.items())[:10]))
        y = list(map(lambda y: y[0], list(self.bi_gram_map.items())[:10]))
        plt.rc('font', family='SimHei', size=13)
        plt.barh(range(10), x, height=0.7, color='steelblue', alpha=0.8)  # 从下往上画
        plt.yticks(range(10), y)
        plt.xlabel("频数")
        plt.title("频率前10的词语")
        for x, y in enumerate(x):
            plt.text(y + 0.2, x - 0.1, '%s' % y)

        # 二元词
        plt.figure(3)
        x = list(map(lambda x: x[1], list(self.tri_gram_map.items())[:10]))
        y = list(map(lambda y: y[0], list(self.tri_gram_map.items())[:10]))
        plt.rc('font', family='SimHei', size=13)
        plt.barh(range(10), x, height=0.7, color='steelblue', alpha=0.8)  # 从下往上画
        plt.yticks(range(10), y)
        plt.xlabel("频数")
        plt.title("频率前10的词语")
        for x, y in enumerate(x):
            plt.text(y + 0.2, x - 0.1, '%s' % y)

        plt.show()


if __name__ == "__main__":
    data_tag = "msr"
    data_path = "./seg-data/training/" + data_tag + "_training.utf8"
    data_statics = DataStatics(data_path)
    print(data_statics.count)
