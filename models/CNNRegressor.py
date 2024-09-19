import torch.nn as nn
import torch.nn.functional as F
class CNNRegressor(nn.Module):
    def __init__(self, output_dim=1):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, output_dim)  # 输出维度设置为1或与目标值维度匹配

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x