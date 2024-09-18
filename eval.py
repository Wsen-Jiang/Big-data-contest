import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

from torch.nn.utils.rnn import pad_sequence
from model import *

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

# 自定义collate_fn函数
def collate_fn(batch):
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    # 获取每个序列的长度
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    # 对序列进行填充
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, lengths, labels

# 指定根目录
root_dir = 'Datasetroot'  # 请将此路径替换为您的数据集根目录
data_dirs = ['1', '2', '3', '4']

# 读取所有数据
sequences, labels = load_data_from_directories(root_dir, data_dirs)

# 创建完整的数据集和数据加载器
full_dataset = WaveformDataset(sequences, labels)
batch_size = 32
data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 定义神经网络模型
model = CNNClassifier()

# 加载保存的模型参数
model.load_state_dict(torch.load('best_model.pth'))

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_X, lengths, batch_y in data_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        lengths = lengths.to(device)

        # 调整输入数据的形状
        batch_X = batch_X.permute(0, 2, 1)  # [batch_size, channels, seq_len]

        #outputs = model(batch_X, lengths)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = 100 * correct / total
print('在整个数据集上的准确率为：{:.2f}%'.format(accuracy))
