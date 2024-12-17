import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):
    """
    从IQ信号中提取特征，尽量避免过度下采样。
    假设输入尺寸: (batch_size, 2, seq_length)
    """

    def __init__(self, in_channels=2, out_channels=64, kernel_size=3, embed_dim=256):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, out_channels, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.proj = nn.Linear(out_channels, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, 2, L)
        x = self.conv1(x)  # (B, 32, L)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)  # (B, 64, L)
        x = self.bn2(x)
        x = F.relu(x)
        # 转换为 (B, L, C) 以适配后续RNN的输入
        x = x.transpose(1, 2)  # (B, L, 64)
        x = self.proj(x)  # (B, L, embed_dim)
        x = self.ln(x)
        return x


class Encoder(nn.Module):
    """
    双向LSTM编码器
    输入：已提取特征的序列 (B, L, embed_dim)
    输出：encoder_outputs (B, L, hidden_size*2), (h, c) 为双向LSTM的隐状态
    """

    def __init__(self, input_dim, hidden_size, num_layers=2, dropout=0.1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers,
                            dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: (B, L, input_dim)
        outputs, (h, c) = self.lstm(x)  # outputs: (B, L, 2*hidden_size)
        return outputs, (h, c)


class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention (Additive Attention)
    输入：decoder_state: (B, hidden_size), encoder_outputs: (B, L, 2*hidden_size)
    输出：上下文向量context (B, hidden_size*2) 和注意力权重weights (B, L)
    """

    def __init__(self, dec_hidden_size, enc_hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W_h = nn.Linear(enc_hidden_size * 2, dec_hidden_size)
        self.W_s = nn.Linear(dec_hidden_size, dec_hidden_size)
        self.v = nn.Linear(dec_hidden_size, 1)

    def forward(self, decoder_state, encoder_outputs):
        # decoder_state: (B, dec_hidden_size)
        # encoder_outputs: (B, L, enc_hidden_size*2)
        score = self.v(torch.tanh(self.W_h(encoder_outputs) + self.W_s(decoder_state.unsqueeze(1))))
        # score: (B, L, 1)
        attn_weights = F.softmax(score, dim=1)  # (B, L, 1)
        context = torch.sum(attn_weights * encoder_outputs, dim=1)  # (B, enc_hidden_size*2)
        return context, attn_weights.squeeze(-1)


class Decoder(nn.Module):
    """
    单向LSTM解码器 + Bahdanau Attention
    输入解码器为上一个时间步的预测（或真实值）和当前的隐状态，以及编码器的输出
    """

    def __init__(self, vocab_size, embed_dim, dec_hidden_size, enc_hidden_size, bos_idx, dropout=0.1):
        super(Decoder, self).__init__()
        self.bos_idx = bos_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTMCell(embed_dim + enc_hidden_size * 2, dec_hidden_size)
        self.attention = BahdanauAttention(dec_hidden_size, enc_hidden_size)
        self.fc = nn.Linear(dec_hidden_size + enc_hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, y_prev, hidden_state, cell_state, encoder_outputs):
        # y_prev: (B,) decoder上个时间步输入符号id
        # hidden_state, cell_state: (B, dec_hidden_size)
        # encoder_outputs: (B, L, enc_hidden_size*2)
        embedded = self.embedding(y_prev)  # (B, embed_dim)
        # 使用Attention获取上下文向量
        context, _ = self.attention(hidden_state, encoder_outputs)  # context: (B, 2*enc_hidden_size)
        lstm_input = torch.cat([embedded, context], dim=-1)  # (B, embed_dim+2*enc_hidden_size)
        h, c = self.lstm(lstm_input, (hidden_state, cell_state))
        # 使用解码器输出 + context计算最终预测
        output = torch.cat([h, context], dim=-1)  # (B, dec_hidden_size+2*enc_hidden_size)
        output = self.fc(self.dropout(output))  # (B, vocab_size)
        return output, h, c

    def forward(self, encoder_outputs, h_0, c_0, tgt=None, max_length=450):
        """
        训练时使用Teacher Forcing：
        encoder_outputs: (B, L, enc_hidden_size*2)
        h_0, c_0: 来自encoder的状态
        tgt: (B, T) 目标序列
        max_length: 推理时生成的最大长度
        """
        B = encoder_outputs.size(0)
        if tgt is not None:
            T = tgt.size(1)
        else:
            T = max_length

        # 解码循环
        outputs = []
        h, c = h_0, c_0
        # 第一步输入Bos token
        y_prev = torch.full((B,), self.bos_idx, dtype=torch.long, device=encoder_outputs.device)

        for t in range(T):
            # 前向一步
            out, h, c = self.forward_step(y_prev, h, c, encoder_outputs)
            outputs.append(out.unsqueeze(1))  # (B, 1, vocab_size)

            if tgt is not None:
                # Training: Teacher Forcing
                y_prev = tgt[:, t]
            else:
                # Inference: Greedy decoding
                y_prev = out.argmax(dim=-1)

        outputs = torch.cat(outputs, dim=1)  # (B, T, vocab_size)
        return outputs


class CQ_LSTM(nn.Module):
    """
    整合CNN特征提取 + 双向LSTM编码器 + Attention解码器
    """

    def __init__(self, vocab_size, bos_idx, embed_dim=256, enc_hidden_size=256, dec_hidden_size=256):
        super(CQ_LSTM, self).__init__()
        self.bos_idx = bos_idx
        self.cnn = CNNFeatureExtractor(in_channels=2, out_channels=64, kernel_size=3, embed_dim=embed_dim)
        self.encoder = Encoder(input_dim=embed_dim, hidden_size=enc_hidden_size, num_layers=2, dropout=0.1)
        self.decoder = Decoder(vocab_size, embed_dim, dec_hidden_size, enc_hidden_size, bos_idx, dropout=0.1)
        # 将encoder最后的双向状态映射到decoder初始状态
        self.enc_to_dec_h = nn.Linear(enc_hidden_size * 2, dec_hidden_size)
        self.enc_to_dec_c = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, src, tgt=None, max_length=450):
        """
        src: (B, 2, L) IQ信号输入
        tgt: (B, T) 码序列目标
        """
        # 1. CNN特征提取
        features = self.cnn(src)  # (B, L, embed_dim)

        # 2. 编码器
        encoder_outputs, (h, c) = self.encoder(features)
        # 双向LSTM的h,c: h,c: (2*num_layers, B, hidden_size)
        # 取最后一层的前向和后向隐状态拼接
        h_forward = h[-2, :, :]
        h_backward = h[-1, :, :]
        c_forward = c[-2, :, :]
        c_backward = c[-1, :, :]

        h_0_dec = torch.tanh(self.enc_to_dec_h(torch.cat([h_forward, h_backward], dim=-1)))  # (B, dec_hidden_size)
        c_0_dec = torch.tanh(self.enc_to_dec_c(torch.cat([c_forward, c_backward], dim=-1)))  # (B, dec_hidden_size)

        # 3. 解码器
        outputs = self.decoder(encoder_outputs, h_0_dec, c_0_dec, tgt, max_length=max_length)
        return outputs

# ===== 使用方法示意 =====
# 假设:
# vocab_size = 12 # 码序列类别数（示例）
# bos_idx = 0     # BOS标记
# model = Seq2SeqModel(vocab_size, bos_idx)
# src = torch.randn(16, 2, 200)    # (batch_size=16, IQ维度=2, 长度=200)
# tgt = torch.randint(1, vocab_size, (16, 50)) # 假设目标序列长度为50
# out = model(src, tgt)  # (B, T, vocab_size)
# loss = F.cross_entropy(out.reshape(-1, vocab_size), tgt.reshape(-1))
# loss.backward()
