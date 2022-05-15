def cut(query, pred):
    sent = query
    tag = pred
    print("sent:", sent)
    print("tag:", tag)
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
    print("words:", words)

    assert len(sent) == len("".join(words)), "还原失败,长度不一致\n{0}\n{1}\n{2}".format(sent, "".join(words),
                                                                                "".join(tag))
    res = "/".join(words)
    return res


if __name__ == "__main__":
    # query = "贾庆林、曾庆红、何鲁丽、肖扬、孙孚凌、万国权以及中央和国家机关、北京市有关部门的负责同志观看了演出。40多个国家的驻华使节及夫人也应邀出席了音乐会。"
    # pred = ['S', 'B', 'E', 'S', 'S', 'B', 'E', 'S', 'S', 'B', 'E', 'S', 'S', 'S', 'S', 'S', 'B', 'E', 'S', 'S', 'B', 'E', 'B', 'E', 'B', 'E', 'S', 'B', 'M', 'M', 'E', 'S', 'B', 'M', 'E', 'B', 'E', 'B', 'E', 'S', 'B', 'E', 'B', 'E', 'B', 'E', 'S', 'B', 'E', 'S', 'B', 'E', 'S', 'S', 'B', 'E', 'S', 'B', 'E', 'B', 'E', 'S', 'B', 'E', 'S', 'B', 'E', 'B', 'E', 'S', 'B', 'M', 'E', 'S']

    query = "2001年1月1日零时，随着新世纪钟声的响起，北京中华世纪坛礼花齐放，万民欢腾。（本报记者徐烨摄）"
    pred = ['B', 'M', 'M', 'M', 'E', 'B', 'E', 'B', 'E', 'B', 'E', 'S', 'B', 'E', 'S', 'B', 'E', 'B', 'E', 'S', 'B', 'E', 'S', 'B', 'E', 'B', 'E', 'B', 'M', 'E', 'B', 'E', 'S', 'S', 'S', 'B', 'E', 'B', 'E', 'S', 'S', 'B', 'E', 'B', 'E', 'S', 'S', 'S', 'S']

    print(cut(query, pred))