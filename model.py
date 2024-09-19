import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier_BN(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNClassifier_BN, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)  # 添加Dropout层
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)  # 应用Dropout
        x = self.fc(x)
        return x

class CNNLSTMClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNLSTMClassifier, self).__init__()

        # CNN 部分
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        # LSTM 部分
        self.lstm = nn.LSTM(input_size=256, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)

        # 全连接层，输入为双向 LSTM 输出（64*2 = 128）
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: [batch_size, 2, seq_len] -> 输入为两路数据（I、Q）
        # CNN 部分
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))  # 输出形状为 [batch_size, 256, seq_len//8]
        # LSTM 期望输入的形状为 (batch_size, seq_len, input_size)
        # 因此我们需要将通道数 (256) 作为 LSTM 的输入特征维度，交换维度顺序
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, 256]

        # LSTM 部分
        lstm_out, _ = self.lstm(x)  # 输出形状为 [batch_size, seq_len//8, 128] （双向 LSTM 的输出）

        # 对所有时间步进行池化，将形状 [batch_size, seq_len, 128] 变为 [batch_size, 128, 1]
        lstm_out = lstm_out.permute(0, 2, 1)  # 交换回通道维度
        lstm_out = self.pool2(lstm_out).squeeze(-1)  # [batch_size, 128]

        # 全连接层进行分类
        out = self.fc(lstm_out)

        return out


class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, channels=2, num_classes=4, dropout=0.1):
        super(CNN_LSTM_Classifier, self).__init__()
        self.conv1 = nn.Conv1d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # LSTM层
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(128, num_classes)

        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        # 卷积和池化
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn3(x)
        x = self.dropout(x)

        # 调整维度以适应LSTM输入 [batch_size, length, channels] -> [batch_size, sequence_length, feature_size]
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

# 定义神经网络模型
class CNN_LSTM_Classifier2(nn.Module):
    def __init__(self, channels=2, num_classes=4, dropout=0.1):
        super(CNN_LSTM_Classifier2, self).__init__()
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

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class CNN_LSTMClassifier(nn.Module):
    def __init__(self, channels=2, num_classes=4, dropout=0.1):  # channels = 2
        super(CNN_LSTMClassifier, self).__init__()
        self.conv1 = nn.Conv1d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # lstm
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        # Linear layer size calculation after LSTM
        # We will set it dynamically later in the forward function
        self.fc1 = nn.Linear(31616, 4096)  # We'll update this to be dynamic
        self.fc2 = nn.Linear(4096, 256)
        self.fc3 = nn.Linear(256, num_classes)  # 修改维度

        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn3(x)
        x = self.dropout(x)
        print(x.shape) # [bz,128,lenth // 8]
        # After convolution, calculate the new length (sequence length)
        x = x.permute(0, 2, 1)  # [batch_size, sequence_length, feature_size]

        # Pass through LSTM
        x, (_, _) = self.lstm(x)

        # Reshape the LSTM output
        x = x.reshape(x.shape[0], -1)  # Flatten all time steps into one feature vector

        # Dynamically calculate the size for fc1 input
        if not hasattr(self, 'fc1'):
            self.fc1 = nn.Linear(x.shape[1], 4096)  # Update fc1 based on dynamic shape

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=4):
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 输入通道数为2（IQ流）
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

def resnet18_1d(num_classes=4):
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_classes=num_classes)


class TransformerClassifier(nn.Module):
    def __init__(self, input_size=2, num_classes=4, d_model=128, nhead=8, num_layers=2, dim_feedforward=512,
                 dropout=0.1):
        super(TransformerClassifier, self).__init__()

        self.input_size = input_size
        self.d_model = d_model

        # 将输入映射到 d_model 维度
        self.input_proj = nn.Linear(input_size, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths):
        # x 的形状: [batch_size, seq_len, input_size]
        # 投影到 d_model 维度
        x = self.input_proj(x)

        # 添加位置编码
        x = self.pos_encoder(x)

        # 生成注意力掩码
        x = x.permute(1, 0, 2)  # 转换为 [seq_len, batch_size, d_model]

        max_len = x.size(0)
        mask = self.generate_padding_mask(lengths, max_len).to(x.device)

        # 通过 Transformer 编码器
        output = self.transformer_encoder(x, src_key_padding_mask=mask)

        # 对时间步求平均
        output = output.mean(dim=0)  # [batch_size, d_model]

        # 分类头
        out = self.fc(output)
        return out

    def generate_padding_mask(self, lengths, max_len):
        device = lengths.device  # 获取 lengths 所在的设备
        mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
        return mask  # 返回形状为 [batch_size, seq_len] 的掩码


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # [batch_size, seq_len, d_model]
        return self.dropout(x)
class S2S_175(nn.Module):
    def __init__(self, input_channel, hidden1, number_classes, dropout):
        super(S2S_175, self).__init__()
        self.conv1 = nn.Conv2d(input_channel,30,11,padding=5)
        self.conv2 = nn.Conv2d(30,30,9,padding=4)
        self.conv3 = nn.Conv2d(30,40,7,padding=3)
        self.conv4 = nn.Conv2d(40,50,5,padding=2)
        self.conv5 = nn.Conv2d(50,50,5,padding=2)
        self.fc = nn.Linear(11200, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.dim_trans = nn.Linear(64*50, 64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.dim_trans(x))
        return x


class Siamese_S2S_175(nn.Module):
    def __init__(self, input_channel, number_classes, window_len, hidden1, hidden2, dropout):
        super(Siamese_S2S_175, self).__init__()
        self.S2S_175 = S2S_175(input_channel, number_classes, hidden1, dropout)
        self.compare_fc = nn.Linear(128, hidden2)  # 修改输入尺寸为200
        self.batch_norm = nn.BatchNorm1d(hidden2)  # 添加批归一化
        self.classifier_fc = nn.Linear(hidden2, 2)  # 两个类别，相似或不相似

    def forward_feature(self, x):
        x = self.S2S_175(x)
        # 展平从 [batch_size, 50, 2] 到 [batch_size, 100]
        # x = x.view(x.size(0), -1)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_feature(input1)
        output2 = self.forward_feature(input2)
        combined = torch.cat((output1, output2), dim=1)  # 沿维度1合并
        combined = self.compare_fc(combined)
        combined = F.relu(self.batch_norm(combined))  # 使用ReLU激活批归一化
        logits = self.classifier_fc(combined)
        return output1, output2, logits


# ------------------------------------------------IQ----------------------------------------------------------
class S2S_IQ(nn.Module):
    def __init__(self, channels, number_classes, window_len, hidden1, dropout):
        super(S2S_IQ, self).__init__()
        self.conv1 = nn.Conv1d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(4608, hidden1)
        self.fc2 = nn.Linear(hidden1, 64)

        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        # x = self.pool(F.ReLU(self.conv1(x)))
        # x = self.bn1(x)
        # x = self.pool(F.ReLU(self.conv2(x)))
        # x = self.bn2(x)
        # x = self.pool(F.ReLU(self.conv3(x)))
        # x = self.bn3(x)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Siamese_S2S_IQ(nn.Module):
    def __init__(self, channels, number_classes, window_len, hidden1, hidden2, dropout):
        super(Siamese_S2S_IQ, self).__init__()
        self.S2S_IQ = S2S_IQ(channels, number_classes, window_len, hidden1, dropout)
        self.compare_fc = nn.Linear(128, hidden2)
        self.batch_norm = nn.BatchNorm1d(hidden2)
        self.classifier_fc = nn.Linear(hidden2, number_classes)

    def forward_feature(self, x):
        x = self.S2S_IQ(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_feature(input1)
        output2 = self.forward_feature(input2)
        combined = torch.cat((output1, output2), dim=1)
        combined = self.compare_fc(combined)
        combined = F.relu(self.batch_norm(combined))
        logits = self.classifier_fc(combined)
        return output1, output2, logits


class S2S_LSTM(nn.Module):
    def __init__(self, channels=2, number_classes=4, hidden_size=32, hidden1=64, dropout=0.5):
        super(S2S_LSTM, self).__init__()
        # 定义卷积层
        self.conv1_1 = nn.Conv1d(channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv1d(channels, 16, kernel_size=5, stride=1, padding=2)
        self.conv1_3 = nn.Conv1d(channels, 16, kernel_size=7, stride=1, padding=3)

        self.conv2 = nn.Conv1d(48, 64, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        # 定义全连接层
        self.fc1 = nn.Linear(hidden_size * 2, hidden1)  # 因为是双向LSTM，所以乘以2
        self.fc2 = nn.Linear(hidden1, number_classes)

    def compute_length_after_conv_pool(self, lengths, conv_kernel_size, conv_stride, conv_padding, pool_kernel_size,
                                       pool_stride, pool_padding):
        # 计算卷积层后的长度
        lengths = ((lengths + 2 * conv_padding - (conv_kernel_size - 1) - 1) // conv_stride) + 1
        # 计算池化层后的长度
        lengths = ((lengths + 2 * pool_padding - (pool_kernel_size - 1) - 1) // pool_stride) + 1
        return lengths

    def forward(self, x, lengths):
        # 输入 x 的形状为 [batch_size, channels, seq_len]

        # 第一层卷积和池化
        x1 = self.pool(F.relu(self.conv1_1(x)))
        x2 = self.pool(F.relu(self.conv1_2(x)))
        x3 = self.pool(F.relu(self.conv1_3(x)))
        x_ = torch.cat((x1, x2, x3), dim=1)  # 在通道维度拼接，dim=1

        # 更新长度
        lengths = self.compute_length_after_conv_pool(lengths, conv_kernel_size=3, conv_stride=1, conv_padding=1,
                                                      pool_kernel_size=3, pool_stride=2, pool_padding=1)

        # 第二层卷积和池化
        x_ = self.pool(F.relu(self.conv2(x_)))
        lengths = self.compute_length_after_conv_pool(lengths, conv_kernel_size=7, conv_stride=1, conv_padding=3,
                                                      pool_kernel_size=3, pool_stride=2, pool_padding=1)

        # 第三层卷积和池化
        x_ = self.pool(F.relu(self.conv3(x_)))
        lengths = self.compute_length_after_conv_pool(lengths, conv_kernel_size=3, conv_stride=1, conv_padding=1,
                                                      pool_kernel_size=3, pool_stride=2, pool_padding=1)

        x_ = self.dropout(x_)
        x_ = x_.permute(0, 2, 1)  # [batch_size, seq_len, features]

        # 确保长度不小于1
        lengths = torch.clamp(lengths, min=1)

        # 使用 pack_padded_sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x_, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        # 获取最后一层的正向和反向的隐状态
        h_n_forward = h_n[-2, :, :]
        h_n_backward = h_n[-1, :, :]
        h_n = torch.cat((h_n_forward, h_n_backward), dim=1)
        x_ = F.relu(self.fc1(h_n))
        x_ = self.fc2(x_)
        return x_


class Siamese_S2S_LSTM(nn.Module):
    def __init__(self, channels, number_classes, window_len, hidden1, hidden2, dropout):
        super(Siamese_S2S_LSTM, self).__init__()
        self.S2S_lstm = S2S_LSTM(channels, number_classes, window_len, hidden1, dropout)
        self.compare_fc = nn.Linear(128, hidden2) # 修改维度
        self.batch_norm = nn.BatchNorm1d(hidden2)
        self.classifier_fc = nn.Linear(hidden2, number_classes)

    def forward_feature(self, x):
        x = self.S2S_lstm(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_feature(input1)
        output2 = self.forward_feature(input2)
        combined = torch.cat((output1, output2), dim=1)
        combined = self.compare_fc(combined)
        combined = F.relu(self.batch_norm(combined))
        logits = self.classifier_fc(combined)
        return output1, output2, logits

class Transformer_IQ(nn.Module):
    def __init__(self, channels, number_classes, window_len, hidden1, dropout):
        super(Transformer_IQ, self).__init__()
        self.conv1 = nn.Conv1d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128,nhead=4,batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer,num_layers=4)

        # self.fc1 = nn.Linear(128, hidden1)
        # self.fc2 = nn.Linear(hidden1, 64)
        self.fc = nn.Linear(128, 64)

        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn3(x)
        
        x = self.dropout(x) # [bz,128,len]
        x = x.permute(0,2,1)# [bz,len,128]
        x = self.transformer(x)
        x = x.mean(dim=1) # 均值池化
        x = self.fc(x)
        return x

class Siamese_S2S_Transformer(nn.Module):
    def __init__(self, channels, number_classes, window_len, hidden1, hidden2, dropout):
        super(Siamese_S2S_Transformer, self).__init__()
        self.transformer_iq = Transformer_IQ(channels, number_classes, window_len, hidden1, dropout)
        self.compare_fc = nn.Linear(128, hidden2)
        self.batch_norm = nn.BatchNorm1d(hidden2)
        self.classifier_fc = nn.Linear(hidden2, number_classes)

    def forward_feature(self, x):
        x = self.transformer_iq(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_feature(input1)
        output2 = self.forward_feature(input2)
        combined = torch.cat((output1, output2), dim=1)
        combined = self.compare_fc(combined)
        combined = F.relu(self.batch_norm(combined))
        logits = self.classifier_fc(combined)
        return output1, output2, logits
