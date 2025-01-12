import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ----------------------------
# 1. Conv1DResidualBlock
# ----------------------------
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

# ----------------------------
# 3. BahdanauAttention
# ----------------------------
class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention (Additive Attention)
    公式参考：
      score(s_t, h_i) = v^T * tanh(W_s * s_t + W_h * h_i)
    """

    def __init__(self, enc_hidden_dim, dec_hidden_dim, attn_dim=128):
        super(BahdanauAttention, self).__init__()
        self.W_s = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        self.W_h = nn.Linear(enc_hidden_dim * 2, attn_dim, bias=False)  # Bi-LSTM输出是2*hidden_dim
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        decoder_hidden: (batch_size, dec_hidden_dim)
        encoder_outputs: (batch_size, seq_len, enc_hidden_dim*2)
        mask: (batch_size, seq_len), 为True的地方表示padding，需要忽略
        """
        # decoder_hidden => (batch_size, 1, dec_hidden_dim)
        dec_hidden_expanded = decoder_hidden.unsqueeze(1)

        # 对decoder_hidden和encoder_outputs做线性变换再相加
        score = self.v(
            torch.tanh(
                self.W_s(dec_hidden_expanded) + self.W_h(encoder_outputs)
            )
        ).squeeze(-1)  # => (batch_size, seq_len)

        # 若有mask，则将padding部分的score设为极小值
        if mask is not None:
            score = score.masked_fill(mask, float('-inf'))

        # 归一化得到注意力权重
        attn_weights = F.softmax(score, dim=-1)  # (batch_size, seq_len)

        # 使用注意力权重对 encoder_outputs 做加权求和
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, enc_hidden_dim*2)
        context = context.squeeze(1)  # (batch_size, enc_hidden_dim*2)

        return context, attn_weights

# ----------------------------
# 4. EncoderCNNBiLSTM
# ----------------------------
class EncoderCNNBiLSTM(nn.Module):
    """
    编码器：先过CNN，再过双向LSTM
    输入: (batch_size, 2, seq_len)  -- 2表示IQ两路
    输出:
        encoder_outputs: (batch_size, seq_len, hidden_dim*2)
        (hidden, cell): 双向LSTM最终的隐藏/细胞状态，可供解码器初始化
    """

    def __init__(self,
                 input_channels=2,
                 cnn_out_channels=64,
                 kernel_size=3,
                 hidden_dim=128,
                 num_layers=2,
                 dropout=0.1):
        super(EncoderCNNBiLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, cnn_out_channels, kernel_size, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

    def forward(self, src, src_lengths=None):
        """
        src: (batch_size, 2, seq_len)
        src_lengths: (batch_size,) 表示每个样本的有效长度(可选, 用于pack_padded_sequence)
        """
        # CNN 提取 => (batch_size, cnn_out_channels, seq_len)
        features = self.cnn(src)

        # 调整维度，以便给LSTM (batch_size, seq_len, cnn_out_channels)
        features = features.permute(0, 2, 1)

        if src_lengths is not None:
            packed_features = nn.utils.rnn.pack_padded_sequence(
                features,
                lengths=src_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            packed_outputs, (hidden, cell) = self.lstm(packed_features)
            encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            encoder_outputs, (hidden, cell) = self.lstm(features)

        # encoder_outputs => (batch_size, seq_len, hidden_dim*2)
        # hidden => (num_layers*2, batch_size, hidden_dim)
        # cell   => (num_layers*2, batch_size, hidden_dim)
        return encoder_outputs, (hidden, cell)

# ----------------------------
# 5. DecoderAttnLSTM
# ----------------------------
class DecoderAttnLSTM(nn.Module):
    """
    解码器：单向LSTM + Bahdanau Attention
    """

    def __init__(self, vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim, num_layers=2, dropout=0.1):
        super(DecoderAttnLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim + enc_hidden_dim * 2,  # 拼接 [当前词嵌入, 上一步context]
            hidden_size=dec_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        self.attention = BahdanauAttention(enc_hidden_dim, dec_hidden_dim)
        self.fc_out = nn.Linear(dec_hidden_dim, vocab_size)

    def forward_step(self, input_tokens, last_hidden, last_cell, encoder_outputs, mask=None):
        """
        只解码一步
        input_tokens: (batch_size,) 当前步输入的token
        last_hidden, last_cell: (num_layers, batch_size, dec_hidden_dim)
        encoder_outputs: (batch_size, seq_len, enc_hidden_dim*2)
        """
        batch_size = input_tokens.size(0)
        # 嵌入 token => (batch_size, embed_dim)
        embedded = self.embedding(input_tokens)

        # Attention: 使用 decoder 最后一层隐藏状态计算注意力
        context, attn_weights = self.attention(
            decoder_hidden=last_hidden[-1],
            encoder_outputs=encoder_outputs,
            mask=mask
        )

        # 拼接 [embedded, context] => (batch_size, embed_dim + enc_hidden_dim*2)
        lstm_input = torch.cat([embedded, context], dim=1).unsqueeze(1)
        output, (hidden, cell) = self.lstm(lstm_input, (last_hidden, last_cell))
        # output => (batch_size, 1, dec_hidden_dim)

        # 计算每个词的输出分布
        output_step = self.fc_out(output.squeeze(1))  # => (batch_size, vocab_size)

        return output_step, hidden, cell, attn_weights

    def forward(self, tgt, encoder_outputs, hidden, cell, teacher_forcing_ratio=1.0, mask=None, bos_idx=0):
        """
        用于训练阶段：一次性解码整条序列
        tgt: (batch_size, tgt_seq_len) 目标序列
        encoder_outputs: (batch_size, src_seq_len, enc_hidden_dim*2)
        hidden, cell: 初始的 decoder hidden/cell
        """
        batch_size, tgt_seq_len = tgt.size()
        outputs = torch.zeros(batch_size, tgt_seq_len, self.fc_out.out_features, device=tgt.device)

        input_token = tgt[:, 0]  # 通常第0个是<BOS>
        for t in range(tgt_seq_len - 1):
            output_step, hidden, cell, attn_weights = self.forward_step(
                input_tokens=input_token,
                last_hidden=hidden,
                last_cell=cell,
                encoder_outputs=encoder_outputs,
                mask=mask
            )
            outputs[:, t, :] = output_step

            # 预测下一个token
            pred_token = output_step.argmax(dim=-1)
            # 决定下一个time step的输入
            teacher_force = random.random() < teacher_forcing_ratio
            input_token = tgt[:, t + 1] if teacher_force else pred_token

        return outputs

# ----------------------------
# 6. IntegratedEncoder
# ----------------------------
class IntegratedEncoder(nn.Module):
    """
    集成了ResBlock的编码器：ResBlock -> CNN -> BiLSTM
    """
    def __init__(self,
                 input_channels=2,
                 res_in_channels=2,
                 res_out_channels=32,
                 res_kernel_size=3,
                 cnn_out_channels=64,
                 cnn_kernel_size=3,
                 hidden_dim=128,
                 num_layers=2,
                 dropout=0.1,
                 use_batchnorm=True):
        super(IntegratedEncoder, self).__init__()

        # 残差块用于初步特征提取
        self.res_block = Conv1DResidualBlock(
            in_channels=res_in_channels,
            out_channels=res_out_channels,
            kernel_size=res_kernel_size,
            stride=1,
            padding=None,
            dilation=1,
            use_batchnorm=use_batchnorm,
            activation=nn.ReLU
        )

        # CNN 用于进一步特征提取
        self.cnn = nn.Sequential(
            nn.Conv1d(res_out_channels, cnn_out_channels, kernel_size=cnn_kernel_size, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU()
        )

        # BiLSTM 编码器
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

    def forward(self, src, src_lengths=None):
        """
        前向传播
        src: (batch_size, 2, seq_len)
        src_lengths: (batch_size,)
        """
        # 通过残差块
        res_out = self.res_block(src)  # (batch_size, res_out_channels, seq_len)

        # 通过CNN
        cnn_out = self.cnn(res_out)  # (batch_size, cnn_out_channels, seq_len)

        # 调整维度以输入LSTM
        features = cnn_out.permute(0, 2, 1)  # (batch_size, seq_len, cnn_out_channels)

        if src_lengths is not None:
            packed_features = nn.utils.rnn.pack_padded_sequence(
                features,
                lengths=src_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            packed_outputs, (hidden, cell) = self.lstm(packed_features)
            encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            encoder_outputs, (hidden, cell) = self.lstm(features)

        # encoder_outputs => (batch_size, seq_len, hidden_dim*2)
        # hidden => (num_layers*2, batch_size, hidden_dim)
        # cell   => (num_layers*2, batch_size, hidden_dim)
        return encoder_outputs, (hidden, cell)

# ----------------------------
# 7. ResBlock_CodeSequence with Correct Hidden State Mapping
# ----------------------------
class ResBlock_CodeSequence(nn.Module):
    """
    集成了ResBlock的整体模型：ResBlock -> CNN -> BiLSTM -> Attention Decoder
    """
    def __init__(self,
                 vocab_size,
                 bos_idx,
                 enc_hidden_dim=128,
                 dec_hidden_dim=128,
                 embed_dim=128,
                 res_in_channels=2,
                 res_out_channels=32,
                 res_kernel_size=3,
                 cnn_out_channels=64,
                 enc_layers=2,
                 dec_layers=2,
                 dropout=0.1,
                 use_batchnorm=True):
        super(ResBlock_CodeSequence, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = IntegratedEncoder(
            input_channels=2,
            res_in_channels=res_in_channels,
            res_out_channels=res_out_channels,
            res_kernel_size=res_kernel_size,
            cnn_out_channels=cnn_out_channels,
            cnn_kernel_size=3,
            hidden_dim=enc_hidden_dim,
            num_layers=enc_layers,
            dropout=dropout,
            use_batchnorm=use_batchnorm
        )
        self.decoder = DecoderAttnLSTM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            enc_hidden_dim=enc_hidden_dim,
            dec_hidden_dim=dec_hidden_dim,
            num_layers=dec_layers,
            dropout=dropout
        )
        # 添加隐藏状态映射层
        self.hidden_map = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.cell_map = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)

    def forward(self, src, src_lengths, tgt=None, teacher_forcing_ratio=1.0):
        """
        前向传播
        src: (batch_size, 2, src_seq_len)
        src_lengths: (batch_size,)
        tgt: (batch_size, tgt_seq_len) 或 None
        """
        # 编码器
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)

        # 转换编码器的隐藏状态以匹配解码器
        num_directions = 2  # BiLSTM
        num_layers = self.encoder.lstm.num_layers
        dec_layers = self.decoder.lstm.num_layers
        dec_hidden_dim = self.decoder.lstm.hidden_size

        # 将BiLSTM的隐藏状态进行线性变换以匹配解码器的隐藏状态
        # hidden: (num_layers * num_directions, batch_size, enc_hidden_dim)
        hidden = hidden.view(num_layers, num_directions, src.size(0), -1)  # (num_layers, 2, batch_size, enc_hidden_dim)
        hidden = torch.cat([hidden[:,0,:,:], hidden[:,1,:,:]], dim=-1)  # (num_layers, batch_size, enc_hidden_dim * 2)
        hidden = self.hidden_map(hidden)  # (num_layers, batch_size, dec_hidden_dim)

        cell = cell.view(num_layers, num_directions, src.size(0), -1)  # (num_layers, 2, batch_size, enc_hidden_dim)
        cell = torch.cat([cell[:,0,:,:], cell[:,1,:,:]], dim=-1)  # (num_layers, batch_size, enc_hidden_dim * 2)
        cell = self.cell_map(cell)  # (num_layers, batch_size, dec_hidden_dim)

        if tgt is not None:
            # 构造mask
            batch_size, max_src_len = encoder_outputs.size(0), encoder_outputs.size(1)
            mask = torch.arange(max_src_len, device=src.device).unsqueeze(0).expand(batch_size, max_src_len)
            mask = mask >= src_lengths.unsqueeze(1)  # True表示padding位置

            outputs = self.decoder(
                tgt=tgt,
                encoder_outputs=encoder_outputs,
                hidden=hidden,
                cell=cell,
                teacher_forcing_ratio=teacher_forcing_ratio,
                mask=mask,
                bos_idx=self.bos_idx
            )
            return outputs  # (batch_size, tgt_seq_len, vocab_size)
        else:
            # 推理阶段
            return self.inference(encoder_outputs, hidden, cell, src_lengths)

    def inference(self, encoder_outputs, hidden, cell, src_lengths, max_decode_len=450):
        """
        推理阶段（不使用teacher forcing），逐步生成token
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        max_src_len = encoder_outputs.size(1)
        mask = torch.arange(max_src_len, device=device).unsqueeze(0).expand(batch_size, max_src_len)
        mask = mask >= src_lengths.unsqueeze(1)

        # 初始化输出序列：起始<BOS>
        generated_tokens = torch.full((batch_size, 1), self.bos_idx, dtype=torch.long, device=device)

        input_token = generated_tokens[:, -1]  # <BOS>
        for _ in range(max_decode_len - 1):
            output_step, hidden, cell, attn_weights = self.decoder.forward_step(
                input_tokens=input_token,
                last_hidden=hidden,
                last_cell=cell,
                encoder_outputs=encoder_outputs,
                mask=mask
            )
            pred_token = output_step.argmax(dim=-1)  # (batch_size,)
            generated_tokens = torch.cat([generated_tokens, pred_token.unsqueeze(1)], dim=1)
            input_token = pred_token

        return generated_tokens  # (batch_size, decoded_length)
