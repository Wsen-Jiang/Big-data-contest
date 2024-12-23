import torch
import torch.nn as nn
import torch.nn.functional as F
import random


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
        # 扩展 decoder_hidden 以便与 encoder_outputs 拼接计算
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
                 num_layers=1,
                 dropout=0.1):
        super(EncoderCNNBiLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, cnn_out_channels, kernel_size, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU()
            # 不做下采样，以免丢失时序信息
            # 需要可选的话可在这里再加几层Conv
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
        # CNN 提取
        # => (batch_size, cnn_out_channels, seq_len)
        features = self.cnn(src)

        # 调整维度，以便给LSTM (batch_size, seq_len, cnn_out_channels)
        features = features.permute(0, 2, 1)

        # 若需要Pack则可使用 pack_padded_sequence
        # 不过要先对序列按长度降序排好序并在inference后再恢复顺序
        # 这里略去，直接送入LSTM
        if src_lengths is not None:
            # pack
            packed_features = nn.utils.rnn.pack_padded_sequence(
                features,
                lengths=src_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            packed_outputs, (hidden, cell) = self.lstm(packed_features)
            # unpack
            encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            encoder_outputs, (hidden, cell) = self.lstm(features)

        # encoder_outputs => (batch_size, seq_len, hidden_dim*2)
        # hidden => (num_layers*2, batch_size, hidden_dim)
        # cell => (num_layers*2, batch_size, hidden_dim)
        return encoder_outputs, (hidden, cell)


class DecoderAttnLSTM(nn.Module):
    """
    解码器：单向LSTM + Bahdanau Attention
    """

    def __init__(self, vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim, num_layers=1, dropout=0.1):
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
        mask: (batch_size, seq_len) 对encoder输出的padding部分做mask
        """
        batch_size = input_tokens.size(0)
        # 嵌入 token
        embedded = self.embedding(input_tokens)  # (batch_size, embed_dim)

        # Attention: 使用 decoder 的上一步 hidden 来计算注意力
        context, attn_weights = self.attention(
            decoder_hidden=last_hidden[-1],  # 取最后一层隐藏状态 (batch_size, dec_hidden_dim)
            encoder_outputs=encoder_outputs,
            mask=mask
        )  # context: (batch_size, enc_hidden_dim*2)

        # 拼接 [embedded, context] 作为 LSTM 的输入
        lstm_input = torch.cat([embedded, context], dim=1).unsqueeze(
            1)  # => (batch_size, 1, embed_dim + enc_hidden_dim*2)
        output, (hidden, cell) = self.lstm(lstm_input, (last_hidden, last_cell))
        # output => (batch_size, 1, dec_hidden_dim)
        # hidden => (num_layers, batch_size, dec_hidden_dim)
        # cell   => (num_layers, batch_size, dec_hidden_dim)

        # 计算每个词的输出分布
        output_step = self.fc_out(output.squeeze(1))  # => (batch_size, vocab_size)

        return output_step, hidden, cell, attn_weights

    def forward(self, tgt, encoder_outputs, hidden, cell, teacher_forcing_ratio=1.0, mask=None, bos_idx=0):
        """
        用于训练阶段：一次性解码整条序列
        tgt: (batch_size, tgt_seq_len) 目标序列（含 <BOS>, <EOS> 等）
        encoder_outputs: (batch_size, src_seq_len, enc_hidden_dim*2)
        hidden, cell: 编码器输出的隐藏状态，可能需要做拼接或变换
        teacher_forcing_ratio: 以真实token作为下一个输入的概率
        mask: (batch_size, src_seq_len)
        bos_idx: <BOS> 的token索引
        """
        batch_size, tgt_seq_len = tgt.size()
        outputs = torch.zeros(batch_size, tgt_seq_len, self.fc_out.out_features, device=tgt.device)

        input_token = tgt[:, 0]  # 第0个token通常是<BOS>
        for t in range(1, tgt_seq_len):
            output_step, hidden, cell, attn_weights = self.forward_step(
                input_tokens=input_token,
                last_hidden=hidden,
                last_cell=cell,
                encoder_outputs=encoder_outputs,
                mask=mask
            )
            outputs[:, t, :] = output_step

            # 预测下一个token
            pred_token = output_step.argmax(dim=-1)  # (batch_size,)

            # 决定下一个time step的输入
            teacher_force = random.random() < teacher_forcing_ratio
            input_token = tgt[:, t] if teacher_force else pred_token

        return outputs  # (batch_size, tgt_seq_len, vocab_size)


class CQ_CNNLSTMAttention(nn.Module):
    """
    组合Encoder + Decoder的完整Seq2Seq
    """

    def __init__(self,
                 vocab_size,
                 bos_idx,
                 enc_hidden_dim=128,
                 dec_hidden_dim=128,
                 embed_dim=128,
                 cnn_out_channels=64,
                 enc_layers=1,
                 dec_layers=1,
                 dropout=0.1):
        super(CQ_CNNLSTMAttention, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = EncoderCNNBiLSTM(
            input_channels=2,
            cnn_out_channels=cnn_out_channels,
            hidden_dim=enc_hidden_dim,
            num_layers=enc_layers,
            dropout=dropout
        )
        self.decoder = DecoderAttnLSTM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            enc_hidden_dim=enc_hidden_dim,
            dec_hidden_dim=dec_hidden_dim,
            num_layers=dec_layers,
            dropout=dropout
        )

    def forward(self, src, src_lengths, tgt=None, teacher_forcing_ratio=1.0):
        """
        训练/推理的统一入口
        src: (batch_size, 2, src_seq_len)
        src_lengths: (batch_size,) 源序列长度
        tgt: (batch_size, tgt_seq_len), 若不为None表示训练/验证
        teacher_forcing_ratio: float
        """
        # 1. 编码器
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)

        # 由于Encoder是双向LSTM，hidden shape是 (num_layers*2, batch_size, enc_hidden_dim)
        # Decoder是单向LSTM，需要做一个简单的转换：把双向的hidden拼接或拆开后再投影
        # 这里直接取 forward 和 backward 的 hidden 叠加或拼接，然后过一个线性变换
        # 为了简单，取最后一层 forward hidden & backward hidden 的平均
        # hidden[ (enc_layers*2 - 2) : (enc_layers*2), ... ] => 取最后一层的 forward/backward
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        dec_init_hidden = torch.mean(torch.stack((forward_hidden, backward_hidden)), dim=0, keepdim=True)
        # cell 同理
        forward_cell = cell[-2, :, :]
        backward_cell = cell[-1, :, :]
        dec_init_cell = torch.mean(torch.stack((forward_cell, backward_cell)), dim=0, keepdim=True)

        # 如果有 tgt，说明是训练或验证阶段
        if tgt is not None:
            # 这里可以构造encoder_output mask, 用于attention忽略padding
            # 若已pack, length < max_len 对应的padding则mask掉
            # for demonstration: mask 只是一个可选流程
            batch_size, max_src_len = encoder_outputs.size(0), encoder_outputs.size(1)
            # 每个样本对应[有效长度, 后面padding], True表示padding要忽略
            mask = torch.arange(max_src_len, device=src.device).unsqueeze(0).expand(batch_size, max_src_len)
            mask = mask >= src_lengths.unsqueeze(1)  # (batch_size, max_src_len)

            outputs = self.decoder(
                tgt=tgt,
                encoder_outputs=encoder_outputs,
                hidden=dec_init_hidden,
                cell=dec_init_cell,
                teacher_forcing_ratio=teacher_forcing_ratio,
                mask=mask,
                bos_idx=self.bos_idx
            )
            return outputs  # (batch_size, tgt_seq_len, vocab_size)
        else:
            # 推理阶段：逐步生成
            return self.inference(encoder_outputs, dec_init_hidden, dec_init_cell, src_lengths)

    def inference(self, encoder_outputs, hidden, cell, src_lengths, max_decode_len=200):
        """
        推理阶段（不使用teacher forcing），逐步生成token
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        # mask
        max_src_len = encoder_outputs.size(1)
        mask = torch.arange(max_src_len, device=device).unsqueeze(0).expand(batch_size, max_src_len)
        mask = mask >= src_lengths.unsqueeze(1)

        # 初始化输出序列：起始<BOS>
        generated_tokens = torch.full((batch_size, 1), self.bos_idx, dtype=torch.long, device=device)

        input_token = generated_tokens[:, -1]  # 刚开始就是<BOS>
        for t in range(max_decode_len - 1):
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
