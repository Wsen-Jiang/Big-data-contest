import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        - x: [seq_length, batch, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class IQToCodeTransformer(nn.Module):
    def __init__(self,
                 input_channels=2,
                 iq_length=1024,
                 max_code_length=450,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dim_feedforward=512,
                 dropout=0.1):
        super(IQToCodeTransformer, self).__init__()
        self.max_code_length = max_code_length
        self.d_model = d_model

        # 特征提取层：使用卷积神经网络提取IQ流特征
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, d_model, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)  # 将长度减半

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, iq_length // 2)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_code_length)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 线性层将Transformer输出映射到二进制概率
        self.output_layer = nn.Linear(d_model, 1)  # 每个位置输出一个概率

    def forward(self, src, tgt):
        """
        - src: [batch_size, 2, iq_length]
        - tgt: [batch_size, max_code_length]
        """
        batch_size = src.size(0)

        # 特征提取
        x = self.relu(self.conv1(src))  # [batch, 64, iq_length]
        x = self.pool(x)  # [batch, 64, iq_length/2]
        x = self.relu(self.conv2(x))  # [batch, d_model, iq_length/2]
        x = x.permute(2, 0, 1)  # [iq_length/2, batch, d_model]

        # 添加位置编码
        x = self.pos_encoder(x)  # [iq_length/2, batch, d_model]

        # Transformer Encoder
        memory = self.transformer_encoder(x)  # [iq_length/2, batch, d_model]

        # 准备解码器输入：使用目标序列的前一部分作为输入（教师强制）
        # 将 tgt 进行映射并添加维度
        tgt_emb = (tgt * 2 - 1).unsqueeze(-1)  # [batch_size, max_code_length, 1]

        # 创建掩码，填充位置为0，其他位置为1
        mask = (tgt != 2).float().unsqueeze(-1)  # [batch_size, max_code_length, 1]

        # 应用掩码
        tgt_emb = tgt_emb * mask  # [batch_size, max_code_length, 1]

        # 调整维度以符合 Transformer 的输入要求
        tgt_emb = tgt_emb.permute(1, 0, 2)  # [max_code_length, batch_size, 1]
        tgt_emb = tgt_emb.repeat(1, 1, self.d_model)  # [max_code_length, batch_size, d_model]

        # 添加位置编码
        tgt_emb = self.pos_decoder(tgt_emb)  # [max_code_length, batch_size, d_model]

        # Transformer Decoder
        output = self.transformer_decoder(tgt_emb, memory)  # [max_code_length, batch_size, d_model]

        # 输出层
        logits = self.output_layer(output)  # [max_code_length, batch_size, 1]
        probs = torch.sigmoid(logits).squeeze(-1).permute(1, 0)  # [batch_size, max_code_length]
        return probs

    def infer(self, iq_sample, device='cuda'):
        """
        对单个IQ样本进行推理
        - iq_sample: [2, iq_length] Tensor
        - device: 设备
        返回：
        - binary_code: [max_code_length] Numpy数组
        """
        iq = iq_sample.unsqueeze(0).to(device)  # [1, 2, iq_length]
        # 创建目标输入，全零
        tgt = torch.ones(1, self.max_code_length).to(device)  # [1, max_code_length]
        y_pred = self.forward(iq, tgt)  # [1, max_code_length]
        y_pred_binary = (y_pred > 0.5).float().squeeze(0).cpu().numpy()  # [max_code_length]
        return y_pred_binary
