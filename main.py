import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
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


# 指定根目录
root_dir = './Dataset'  # 请将此路径替换为您的数据集根目录
data_dirs = ['1', '2', '3', '4']

# 读取数据
sequences, labels = load_data_from_directories(root_dir, data_dirs)

# 划分训练集和验证集
seq_train, seq_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# 创建数据集和数据加载器
train_dataset = WaveformDataset(seq_train, y_train)
val_dataset = WaveformDataset(seq_val, y_val)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)


# 定义神经网络模型


model = resnet18_1d()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

best_accuracy = 0

for epoch in range(num_epochs):
    model.train()
    for batch_X, lengths, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        lengths = lengths.to(device)

        # 调整输入数据的形状
        batch_X = batch_X.permute(0, 2, 1)  # [batch_size, channels, seq_len]

        optimizer.zero_grad()
        #outputs = model(batch_X, lengths)
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # 验证模型
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, lengths, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            lengths = lengths.to(device)

            batch_X = batch_X.permute(0, 2, 1)  # [batch_size, channels, seq_len]

            #outputs = model(batch_X, lengths)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    print('Epoch [{}/{}], Val Loss: {:.4f}, Accuracy: {:.2f}%'.format(
        epoch + 1, num_epochs, val_loss, accuracy))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model.pth')
torch.save(model.state_dict(), 'final_model.pth')