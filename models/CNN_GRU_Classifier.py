import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_GRU_Classifier(nn.Module):
    def __init__(self, channels=2, num_classes=10, dropout=0.2):
        super(CNN_GRU_Classifier, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Conv1d(channels, 64, kernel_size=3, stride=1, padding=1)  # 输出通道改为 64
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)

        # 第二层卷积
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3)  # 输入改为 64，输出改为 128
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        # 第三层卷积
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)  # 输入改为 128，输出改为 256
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)

        # 自适应池化
        self.pool = nn.AdaptiveAvgPool1d(output_size=100)

        # GRU 替代 LSTM
        self.rnn = nn.GRU(input_size=256, hidden_size=128, num_layers=2, batch_first=True,
                          bidirectional=True)  # 输入改为 256

        # Attention 层
        self.attention = SelfAttention(256)  # 输入改为 GRU 输出的维度：2 * hidden_size = 256

        # 全连接层
        self.fc1 = nn.Linear(256, num_classes)

        # Dropout
        self.dropout_final = nn.Dropout(dropout)

    def forward(self, x):
        # 第一层卷积
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        # 第二层卷积
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        # 第三层卷积
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)

        # 调整维度以适配 GRU 输入
        x = x.permute(0, 2, 1)

        # GRU 输出
        x, _ = self.rnn(x)

        # Attention 层
        x = self.attention(x)

        # Dropout 和全连接层
        x = self.dropout_final(x)
        x = self.fc1(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x):
        weights = F.softmax(self.attention(x), dim=1)  # 注意力权重
        x = torch.sum(x * weights, dim=1)  # 加权求和
        return x
