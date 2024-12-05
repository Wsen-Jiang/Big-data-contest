import torch.nn as nn
import torch
import torch.nn.functional as F
class DeepCNN_LSTMClassifier(nn.Module):
    def __init__(self, channels=2, num_classes=10, dropout=0.1):
        super(DeepCNN_LSTMClassifier, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv1d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3)  # padding=3 保持长度不变
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # LSTM层
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc1 = nn.Linear(64 * 2, 256)  # 双向LSTM，所以隐藏层大小乘以2
        self.fc2 = nn.Linear(256, num_classes)

        # 参数初始化
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, lengths):
        """
        Args:
            x: Tensor of shape [batch_size, channels, seq_len]
            lengths: Tensor of shape [batch_size] 包含每个序列的原始长度
        Returns:
            logits: Tensor of shape [batch_size, num_classes]
        """
        # 卷积层与池化
        x = self.pool(F.relu(self.conv1(x)))  # [batch_size, 32, seq_len/2]
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))  # [batch_size, 64, seq_len/4]
        x = self.bn2(x)
        x = self.pool(F.relu(self.conv3(x)))  # [batch_size, 128, seq_len/8]
        x = self.bn3(x)
        x = self.dropout(x)

        # 调整序列长度
        # 每次池化将长度减半，经过3次池化总共减8倍
        new_lengths = lengths // 8  # 整数除法
        new_lengths = torch.clamp(new_lengths, min=1)  # 确保长度至少为1

        # 调整维度以适应LSTM输入 [batch_size, seq_len, channels]
        x = x.permute(0, 2, 1)  # [batch_size, seq_len/8, 128]

        # 使用 pack_padded_sequence 处理变长序列
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, new_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        packed_output, (h_n, c_n) = self.lstm(packed_input)

        # 获取最后一层的正向和反向的隐藏状态
        h_n_forward = h_n[-2, :, :]  # [batch_size, hidden_size]
        h_n_backward = h_n[-1, :, :]  # [batch_size, hidden_size]
        h_n = torch.cat((h_n_forward, h_n_backward), dim=1)  # [batch_size, 2 * hidden_size]

        h_n = self.dropout(h_n)

        # 全连接层
        x = F.relu(self.fc1(h_n))  # [batch_size, 256]
        logits = self.fc2(x)        # [batch_size, num_classes]
        return logits
