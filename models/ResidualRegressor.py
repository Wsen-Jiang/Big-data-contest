import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip_connection = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if self.skip_connection is not None:
            identity = self.skip_connection(identity)
        out += identity
        return F.relu(out)

class ResidualRegressor(nn.Module):
    def __init__(self, output_dim=1):
        super(ResidualRegressor, self).__init__()
        self.resblock1 = ResidualBlock(2, 64)  # 第一层
        self.resblock2 = ResidualBlock(64, 128)  # 第二层
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, output_dim)  # 输出通道数与最后一层匹配

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

