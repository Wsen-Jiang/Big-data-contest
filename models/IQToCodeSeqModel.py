# models/IQToCodeSeqModel.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class IQToCodeSeqModel(nn.Module):
    def __init__(self, vocab_size, bos_idx, embed_dim=256, num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                 max_seq_length=450, dropout=0.1):
        super(IQToCodeSeqModel, self).__init__()
        self.bos_idx = bos_idx

        # CNN部分：提取IQ流的特征
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # 增加一层线性变换将CNN输出映射到Transformer需要的维度，并加入LayerNorm稳定分布
        self.cnn_proj = nn.Linear(128, embed_dim)
        self.cnn_norm = nn.LayerNorm(embed_dim)

        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_encoder_layers
        )

        # Transformer解码器
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_decoder_layers
        )

        # 码序列嵌入
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 位置编码 (与max_seq_length保持一致或略大，但不必过大)
        self.register_buffer('positional_encoding', self._generate_positional_encoding(embed_dim, 600))

        # 输出层
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.max_seq_length = max_seq_length

    def _generate_positional_encoding(self, embed_dim, max_seq_length):
        position = torch.arange(0, max_seq_length).unsqueeze(1)  # (max_seq_length, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_seq_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_seq_length, embed_dim)

    def forward(self, src, tgt=None):
        """
        Args:
            src: (batch_size, 2, src_seq_length)
            tgt: (batch_size, tgt_seq_length)
        """

        # CNN特征提取
        cnn_out = self.cnn(src)  # (batch_size, 128, src_seq_len/4)
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch_size, src_seq_len/4, 128)

        # 映射到Transformer输入维度并归一化
        cnn_out = self.cnn_proj(cnn_out)
        cnn_out = self.cnn_norm(cnn_out)

        # 添加位置编码
        seq_len = cnn_out.size(1)
        encoder_input = cnn_out + self.positional_encoding[:, :seq_len, :]

        # 编码器
        memory = self.transformer_encoder(encoder_input)

        if tgt is not None:
            # 训练模式
            batch_size, tgt_seq_length = tgt.size()
            bos_tensor = torch.full((batch_size, 1), self.bos_idx, dtype=torch.long, device=src.device)
            tgt_input = torch.cat([bos_tensor, tgt[:, :-1]], dim=1)

            tgt_emb = self.embedding(tgt_input) + self.positional_encoding[:, :tgt_input.size(1), :]

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(src.device)

            decoder_output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.output_layer(decoder_output)
            return logits
        else:
            # 推理模式：逐步生成
            batch_size = src.size(0)
            generated = torch.full((batch_size, 1), self.bos_idx, dtype=torch.long, device=src.device)

            for t in range(self.max_seq_length - 1):
                input_emb = self.embedding(generated) + self.positional_encoding[:, :generated.size(1), :]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1)).to(src.device)
                output = self.transformer_decoder(input_emb, memory, tgt_mask=tgt_mask)
                output_step = self.output_layer(output[:, -1, :])
                next_token = output_step.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

            return generated
