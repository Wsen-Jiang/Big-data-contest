from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

# 自定义collate_fn函数，用于处理变长序列
# 在处理变长序列时，默认的 DataLoader 无法将不同长度的序列组合成一个批次，因此需要自定义 collate_fn 函数来实现填充操作。
def collate_fn(batch):
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    # 获取每个序列的长度
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    # 对序列进行填充
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, lengths, labels


# 定义自定义数据集类
class WaveformDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences  # 序列列表，每个元素是一个Tensor
        self.labels = labels  # 标签列表

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# 数据读取函数
def load_data_from_directories(root_dir, data_dirs):
    sequences = []
    labels = []
    # 遍历每个目录
    for dir_name in data_dirs:
        label = int(dir_name) - 1  # 将标签调整为从0开始
        dir_path = os.path.join(root_dir, dir_name)
        for file_name in os.listdir(dir_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(dir_path, file_name)
                # 读取CSV文件，无标题行
                data = pd.read_excel(file_path, engine='xlrd',header=None)
                # 提取第一列和第二列作为输入特征
                sequence = data.iloc[:, [0, 1]].values.astype(np.float32)
                sequence = torch.tensor(sequence)
                # 添加到列表
                sequences.append(sequence)
                labels.append(label)
    return sequences, labels