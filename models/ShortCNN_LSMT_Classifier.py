import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM_Classifier_L(nn.Module):
    def __init__(self, channels=2, num_classes=4, dropout=0.6):
        super(CNN_LSTM_Classifier_L, self).__init__()
        self.conv1 = nn.Conv1d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)


        # LSTM层
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(64, num_classes)

        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        # 卷积和池化
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.dropout(x)

        # 调整维度以适应LSMT输入 [batch_size, length, channels] -> [batch_size, sequence_length, feature_size]
        x = x.permute(0, 2, 1)

        # LSTM输出
        x, (_, _) = self.lstm(x)
        x = torch.mean(x,dim=1) # [bz,128]

        # 将LSTM的输出展平为 [batch_size, -1]，即把sequence_length和hidden_size合并成一个维度
        # x = x.reshape(x.shape[0], -1)
        # # 动态计算全连接层的输入大小
        # if self.fc1 is None:
        #     self.fc1 = nn.Linear(x.shape[1], 4096)
        #     nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')

        # 全连接层
        x = F.relu(self.fc1(x))
        return x