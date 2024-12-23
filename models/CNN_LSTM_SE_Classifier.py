


import torch
import torch.nn as nn
import torch.nn.functional as F


# fc-变种
class CNN_LSTM_SE_Classifier(nn.Module):
    def __init__(self, channels=2, num_classes=10, dropout=0.5):
        super(CNN_LSTM_SE_Classifier, self).__init__()
        # (Same model as before)
        self.conv1 = nn.Conv1d(channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        # self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()

        # 参数初始化
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x_ = x
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = x * x_
        x = self.fc2(x)
        return x
