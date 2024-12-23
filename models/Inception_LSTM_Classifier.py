

import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule1D(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModule1D, self).__init__()
        # in_channels : 128
        # 1x1 卷积分支
        self.branch1x1 = nn.Conv1d(in_channels, 48, kernel_size=1)
        
        # 1x1 卷积后接 3x3 卷积分支
        self.branch3x3_1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.branch3x3_2 = nn.Conv1d(64, 96, kernel_size=3, padding=1)
        
        # 1x1 卷积后接 5x5 卷积分支
        self.branch5x5_1 = nn.Conv1d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        
        # 3x3 最大池化后接 1x1 卷积分支
        self.branch_pool = nn.Conv1d(in_channels, 32, kernel_size=1)

        nn.init.kaiming_uniform_(self.branch1x1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.branch3x3_1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.branch3x3_2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.branch5x5_1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.branch5x5_2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.branch_pool.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        # 1x1 卷积分支
        branch1x1 = self.branch1x1(x)

        # 1x1 卷积后接 3x3 卷积分支
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        
        # 1x1 卷积后接 5x5 卷积分支
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        
        # 3x3 最大池化后接 1x1 卷积分支
        branch_pool = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # 将所有分支的输出在通道维度上进行拼接
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class Inception_LSTM_Classifier(nn.Module):
    def __init__(self, channels=2, num_classes=10, dropout=0.5):
        super(Inception_LSTM_Classifier, self).__init__()
        # (Same model as before)
        # self.conv1 = nn.Conv1d(channels, 64, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.conv2 = nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        
        self.inception = InceptionModule1D(channels)

        self.lstm = nn.LSTM(input_size=208, hidden_size=104, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(208, num_classes)

        # 参数初始化
        # nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        # x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.inception(x)
        x = self.pool(x)

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x