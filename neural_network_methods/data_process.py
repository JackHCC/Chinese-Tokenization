from codecs import open


def build_corpus(data_tag, split, make_vocab=True, fix_length=-1):
    """读取数据"""
    assert data_tag in ["pku", "msr"]
    assert split in ['train', 'dev', 'test']

    if split in ['train', 'dev']:
        filename = './seg-data/training/' + data_tag + '_training.utf8'
    else:
        filename = './seg-data/gold/' + data_tag + '_test_gold.utf8'

    whole_word_list, word_lists, whole_label_list, tag_lists = read_file(filename)

    if fix_length != -1:
        word_lists = [whole_word_list[i:i + fix_length] for i in range(0, len(whole_word_list), fix_length)]
        tag_lists = [whole_label_list[i:i + fix_length] for i in range(0, len(whole_label_list), fix_length)]

    if split == 'dev':
        word_lists = word_lists[:1000]
        tag_lists = tag_lists[:1000]

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps


# convert line to chars
def get_char(sentence):
    return list(''.join(sentence.split(' ')))


# convert line to B E M S label
def get_label(sentence):
    output_str = []
    word_list = sentence.split()
    for i in range(len(word_list)):
        if len(word_list[i]) == 1:
            output_str.append('S')
        elif len(word_list[i]) == 2:
            output_str.append('B')
            output_str.append('E')
        else:
            M_num = len(word_list[i]) - 2
            output_str.append('B')
            output_str.extend('M' * M_num)
            output_str.append('E')
    return output_str


def read_file(filename):
    char_list, line_list, label_list, label_line_list = [], [], [], []

    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if line != '':
            line_char_list = get_char(line)
            line_label_list = get_label(line)

            char_list.extend(line_char_list)
            line_list.append(line_char_list)
            label_list.extend(line_label_list)
            label_line_list.append(line_label_list)
    return char_list, line_list, label_list, label_line_list
