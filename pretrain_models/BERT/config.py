import os
import torch

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '11221'

bert_model = '/nfs/volume-1280-3/hcc/Learning/Models/bert-base-chinese/'
roberta_model = '/nfs/volume-1280-3/rushin/work/models/hfl/chinese_roberta_wwm_large_ext/'

model_dir = os.getcwd() + '/experiments/bert_pku/'
data_dir = os.getcwd() + '/data/pku/'
train_dir = data_dir + 'training.npz'
test_dir = data_dir + 'test.npz'
# test_dir = data_dir + 'test_case.npz'
files = ['training', 'test']
# files = ['training', 'test_case']
vocab_path = data_dir + 'vocab.npz'
exp_dir = os.getcwd() + '/experiments/bert_msr/'
log_dir = exp_dir + 'train.log'
case_dir = os.getcwd() + '/case/bad_case.txt'
output_dir = data_dir + 'output.txt'
res_dir = data_dir + 'res.txt'
test_ans = data_dir + 'test.txt'

max_vocab_size = 1000000
max_len = 500
sep_word = '@'  # 拆分句子的文本分隔符
sep_label = 'S'  # 拆分句子的标签分隔符

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的Seg模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 1e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 6
epoch_num = 50
min_epoch_num = 5
patience = 0.0002
patience_num = 4

# gpu_num = torch.cuda.device_count()
gpu_num = 5
device_ids = list(range(gpu_num))

gpu = '0'

if gpu != '':
    torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cpu")

# B：分词头部 M：分词词中 E：分词词尾 S：独立成词
label2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}

id2label = {_id: _label for _label, _id in list(label2id.items())}
