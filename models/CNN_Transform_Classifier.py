import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CNN_Transform_Classifier(nn.Module):
    def __init__(self, channels=2, out_dim=1, dropout=0.7):
        super(CNN_Transform_Classifier, self).__init__()
        # 卷积部分
        self.conv1 = nn.Conv1d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # Transformer部分
        self.embedding_dim = 128  # 与最后一个卷积层的输出通道保持一致
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=64,
            dropout=dropout,
            batch_first=True  # 启用 batch_first
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=1)

        # 全连接层
        self.fc = nn.Linear(self.embedding_dim,out_dim)

    def forward(self, x):
        # 卷积层提取局部特征
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))# 输出形状 [batch_size, channels, seq_len]
        x = x.permute(0, 2, 1)  # 转换为 [batch_size, seq_len, features]

        # Transformer提取全局特征
        x = self.transformer(x)  # 输入 [batch_size, seq_len, embedding_dim]
        x = torch.mean(x, dim=1)  # 平均池化，输出 [batch_size, embedding_dim]

        # 全连接层输出分类
        x = self.fc(x)  # 输出 [batch_size, num_classes]
        return x
