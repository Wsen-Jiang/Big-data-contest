import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import torch.nn.functional as F
from model import *
from dataset import load_data_from_directories,WaveformDataset,collate_fn
from utils import show_plot



# 指定根目录
root_dir = 'Dataset'  # 请将此路径替换为您的数据集根目录
data_dirs = ['1', '2', '3', '4']
save_loss_path = "./save_loss/"
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


model = CNNClassifier()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

best_accuracy = 0
history_train_loss = []
history_valid_loss = []
for epoch in range(num_epochs):
    train_loss = 0
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
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(train_loader)
    history_train_loss.append(train_loss)

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
    history_valid_loss.append(val_loss)
    accuracy = 100 * correct / total
    print('Epoch [{}/{}], Val Loss: {:.4f}, Accuracy: {:.2f}%'.format(
        epoch + 1, num_epochs, val_loss, accuracy))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # 检查目录是否存在，如果不存在则创建
        model_dir = f'models/{model.__class__.__name__}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # 然后保存模型
        torch.save(model.state_dict(), os.path.join(model_dir, f'{num_epochs}_best_model.pth'))
torch.save(model.state_dict(), f'models/{model.__class__.__name__}/final_model.pth')

show_plot(history_train_loss, history_valid_loss, save_loss_path + f"acc_{best_accuracy}_{model.__class__.__name__}.png")
