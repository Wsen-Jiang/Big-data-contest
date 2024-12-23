import torch
import torch.nn as nn
import torch.nn.functional as F
import random
class IQToCodeSeqModel_Old(nn.Module):
    def __init__(self, vocab_size, bos_idx, embed_dim=128, num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                 max_seq_length=450, dropout=0.1):
        super(IQToCodeSeqModel_Old, self).__init__()
        self.bos_idx = bos_idx  # 存储 <BOS> 的索引
        # CNN部分
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_encoder_layers
        )
        # Transformer解码器
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=128, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_decoder_layers
        )
        # 码序列嵌入
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(embed_dim, max_seq_length), requires_grad=False)
        # 输出层
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.max_seq_length = max_seq_length
    def _generate_positional_encoding(self, embed_dim, max_seq_length):
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_seq_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # 修改为 (1, max_seq_length, embed_dim)
    def forward(self, src, tgt=None, teacher_forcing_ratio=1.0):
        """
        Args:
            src: 源输入张量，形状为 (batch_size, channels, src_seq_length)
            src_lengths: 源序列长度
            tgt: 目标序列张量，形状为 (batch_size, tgt_seq_length)
            teacher_forcing_ratio: 使用真实标签作为输入的概率
        Returns:
            输出序列的 logits 或生成的序列
        """
        # CNN部分
        cnn_out = self.cnn(src)  # (batch_size, 128, src_seq_length/4)
        cnn_out = cnn_out.permute(0, 2, 1)  # 调整为 (batch_size, src_seq_length/4, 128)
        memory = self.transformer_encoder(cnn_out)  # (batch_size, src_seq_length/4, 128)
        if tgt is not None:
            # 训练阶段：使用教师强制，批量处理整个目标序列
            batch_size, tgt_seq_length = tgt.size()
            # 准备解码器输入：<BOS> + tgt[:, :-1]
            bos_tensor = torch.full((batch_size, 1), self.bos_idx, dtype=torch.long, device=src.device)
            tgt_input = torch.cat([bos_tensor, tgt[:, :-1]], dim=1)  # (batch_size, tgt_seq_length)
            # 嵌入和位置编码
            tgt_emb = self.embedding(tgt_input) + self.positional_encoding[:, :tgt_input.size(1), :].to(
                src.device)  # (batch_size, tgt_seq_length, embed_dim)
            # 生成上三角掩码，确保解码器只能看到当前位置之前的token
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(
                src.device)  # (tgt_seq_length, tgt_seq_length)
            # 解码器
            decoder_output = self.transformer_decoder(tgt_emb, memory,
                                                      tgt_mask=tgt_mask)  # (batch_size, tgt_seq_length, embed_dim)
            # 输出层
            logits = self.output_layer(decoder_output)  # (batch_size, tgt_seq_length, vocab_size)
            return logits  # 在训练时返回logits，用于计算损失
            # # 训练阶段或带有目标序列的推理
            # batch_size, tgt_seq_length = tgt.size()
            # outputs = torch.zeros(batch_size, tgt_seq_length, self.output_layer.out_features).to(src.device)
            #
            # # 初始化输入序列
            # input_seq = tgt[:, 0].unsqueeze(1)  # (batch_size, 1)
            # input_emb = self.embedding(input_seq) + self.positional_encoding[:, :1, :].to(src.device)  # (batch_size, 1, embed_dim)
            #
            # for t in range(tgt_seq_length):
            #     # 解码器
            #     output = self.transformer_decoder(input_emb, memory)  # (batch_size, current_length, embed_dim)
            #     output_step = self.output_layer(output[:, -1, :])  # (batch_size, vocab_size)
            #     outputs[:, t, :] = output_step
            #
            #     # 决定下一个输入
            #     teacher_force = random.random() < teacher_forcing_ratio
            #     top1 = output_step.argmax(dim=-1)  # (batch_size)
            #
            #     input_seq = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)  # (batch_size, 1)
            #     input_emb = self.embedding(input_seq) + self.positional_encoding[:, t+1:t+2, :].to(src.device)  # (batch_size, 1, embed_dim)
            # return outputs  # (batch_size, tgt_seq_length, vocab_size)
        else:
            # 推理阶段：逐步生成
            batch_size = src.size(0)
            generated = torch.full((batch_size, 1), self.bos_idx, dtype=torch.long, device=src.device)  # 初始化为 <BOS> 标记
            for t in range(self.max_seq_length - 1):
                input_emb = self.embedding(generated) + self.positional_encoding[:, :generated.size(1), :].to(src.device)  # (batch_size, current_length, embed_dim)
                output = self.transformer_decoder(input_emb, memory)  # (batch_size, current_length, embed_dim)
                output_step = self.output_layer(output[:, -1, :])  # (batch_size, vocab_size)
                next_tokens = output_step.argmax(dim=-1).unsqueeze(1)  # (batch_size, 1)
                generated = torch.cat([generated, next_tokens], dim=1)  # (batch_size, current_length +1)
            return generated  # 返回生成的索引序列