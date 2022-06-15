import jieba
import pkuseg
import thulac


class SegLibEval:
    def __init__(self, lib_name, data_name):
        self.data_path = "./seg-data/testing/" + data_name + "_test.utf8"
        self.lib_name = lib_name
        self.cut = None
        self.get_cut_method()

        self.cut_results = self.cut_file_text()

    def get_cut_method(self):
        if self.lib_name == "jieba":
            self.cut = jieba.cut
        elif self.lib_name == "pkuseg":
            seg = pkuseg.pkuseg()  # 以默认配置加载模型
            self.cut = seg.cut
        elif self.lib_name == "thu":
            thu1 = thulac.thulac(seg_only=True)
            self.cut = thu1.cut

    def cut_sent(self, sent):
        if self.lib_name == "jieba":
            line_cut = list(self.cut(sent))
        elif self.lib_name == "thu":
            line_cut = [item[0] for item in self.cut(sent)]
        else:
            line_cut = self.cut(sent)
        line_cut_str = "/".join(line_cut)
        return line_cut_str

    def cut_file_text(self):
        results = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if self.lib_name == "jieba":
                    line_cut = list(self.cut(line))
                elif self.lib_name == "thu":
                    line_cut = [item[0] for item in self.cut(line)]
                else:
                    line_cut = self.cut(line)
                if not line_cut:
                    continue
                line_cut_str = " ".join(line_cut)
                results.append(line_cut_str)
        return results

    def save_cut_to_file(self, save_path):
        with open(save_path, "w", encoding="utf-8") as f:
            for line in self.cut_results:
                f.write(line + "\n")


if __name__ == "__main__":
    query = "他一把把把把住了"

    method = "thu"
    data_name = "pku"
    # save_path = "./results/" + method + "_seg_" + data_name + ".txt"

    seg_lib = SegLibEval(method, data_name)
    print(seg_lib.cut_sent(query))

    # 保存分词结果
    # seg_lib.save_cut_to_file(save_path)


