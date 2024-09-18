import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

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

class CNNTransformerClassifier(nn.Module):
    def __init__(self, input_dim=2, num_classes=4, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(CNNTransformerClassifier, self).__init__()
        self.conv = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths):
        #print("x before permute:", x.shape)  # 应该是 [batch_size, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, seq_len]
        #print("x after permute:", x.shape)  # 应该是 [batch_size, input_dim, seq_len]
        x = self.conv(x)  # [batch_size, d_model, seq_len]
        #print("x after conv:", x.shape)  # [batch_size, d_model, seq_len]
        x = x.permute(2, 0, 1)  # [seq_len, batch_size, d_model]
        #print("x after second permute:", x.shape)  # [seq_len, batch_size, d_model]

        # 创建注意力掩码
        max_len = x.size(0)
        mask = torch.arange(max_len).unsqueeze(0).to(lengths.device) >= lengths.unsqueeze(1)
        mask = mask.to(x.device)

        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        output = output.mean(dim=0)
        logits = self.fc(output)
        return logits



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

# class S2S_LSTM(nn.Module):
#     def __init__(self, channels, number_classes, window_len, hidden1, dropout): # channels = 2
#         super(S2S_LSTM, self).__init__()
#         self.conv1 = nn.Conv1d(channels, 32, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm1d(32)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
#         #lstm
#         self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first= True, bidirectional=True) # 双向LSTM
#         self.dropout = nn.Dropout(dropout)

#         self.fc1 = nn.Linear(5376, hidden1)
#         self.fc2 = nn.Linear(hidden1, 64)  # 修改维度


#         nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
#         nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
#         nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
#         nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
#         nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.bn1(x)
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.bn2(x)
#         x = self.pool(F.relu(self.conv3(x)))
#         x = self.bn3(x)
#         x = self.dropout(x)
#         x = x.permute(0, 2, 1) # [bz,len,128]
#         res_x = x
#         x,(_, _) = self.lstm(res_x) # [bz,len,128]
#         x = x + res_x
#         x = x.reshape(x.shape[0],-1) # 全部时间步特征
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


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
