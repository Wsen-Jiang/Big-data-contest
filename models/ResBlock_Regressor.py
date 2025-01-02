import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1, use_batchnorm=True,
                 activation=nn.ReLU):
        """
        初始化 Conv1D 残差块。

        参数：
        - in_channels (int): 输入通道数。
        - out_channels (int): 输出通道数。
        - kernel_size (int): 卷积核大小。默认值为3。
        - stride (int): 卷积步幅。默认值为1。
        - padding (int): 填充大小。如果为 None，将自动计算以保持长度不变。
        - dilation (int): 扩张系数。默认值为1。
        - use_batchnorm (bool): 是否使用批归一化。默认值为True。
        - activation (nn.Module): 激活函数。默认值为ReLU。
        """
        super(Conv1DResidualBlock, self).__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation  # 保持长度不变

        self.use_batchnorm = use_batchnorm
        self.activation = activation()

        # 主路径
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, bias=not use_batchnorm)
        self.bn1 = nn.BatchNorm1d(out_channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1,
                               padding=padding, dilation=dilation, bias=not use_batchnorm)
        self.bn2 = nn.BatchNorm1d(out_channels) if use_batchnorm else nn.Identity()

        # 跳跃连接的投影层（如果需要）
        if in_channels != out_channels or stride != 1:
            self.projection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=not use_batchnorm),
                nn.BatchNorm1d(out_channels) if use_batchnorm else nn.Identity()
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        """
        前向传播。

        参数：
        - x (torch.Tensor): 输入张量，形状为 (batch_size, in_channels, seq_length)。

        返回：
        - torch.Tensor: 输出张量，形状为 (batch_size, out_channels, new_seq_length)。
        """
        identity = self.projection(x)  # 跳跃连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 添加跳跃连接
        out = self.activation(out)

        return out


class ResBlock_Regressor(nn.Module):
    def __init__(self, channels=2, dropout=0.5):
        super(ResBlock_Regressor, self).__init__()
        self.conv1 = nn.Conv1d(channels, 30, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(30)
        self.res1 = Conv1DResidualBlock(32, 32)

        self.conv2 = nn.Conv1d(32, 96, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(96)
        self.res2 = Conv1DResidualBlock(128, 128)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.res3 = Conv1DResidualBlock(256, 256)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        # 全连接层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, 1)  # 修改为单一回归输出

        # 参数初始化
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):  # (bz, 2, 512)
        x1 = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = torch.cat((x1, x), dim=1)  # (bz, 32, 512)
        x = self.res1(x)
        x = self.pool(x)  # (bz, 32, 256)

        x2 = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.cat((x2, x), dim=1)  # (bz, 128, 256)
        x = self.res2(x)
        x = self.pool(x)  # (bz, 128, 128)

        x3 = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.cat((x3, x), dim=1)  # (bz, 256, 128)
        x = self.res3(x)
        x = self.pool(x)  # (bz, 256, 64)

        x = x.permute(0, 2, 1)  # (batch_size, seq_length, features)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)  # 输出回归值
        return x
