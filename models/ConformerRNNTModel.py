import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


################################################################################
# Conformer Block 定义
################################################################################
class FeedForwardModule(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, expansion_factor * dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(expansion_factor * dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return out


class ConformerConvModule(nn.Module):
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                        groups=dim, padding=kernel_size // 2)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.swish = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.pointwise_conv1(x)  # (B, 2D, T)
        x = self.glu(x)  # (B, D, T)
        x = self.depthwise_conv(x)  # (B, D, T)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, T, D)
        return x


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.mha(x, x, x)
        out = self.dropout(out)
        return self.layer_norm(x + out)


class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_expansion_factor=4, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, ff_expansion_factor, dropout)
        self.ff2 = FeedForwardModule(dim, ff_expansion_factor, dropout)
        self.attn = MultiHeadSelfAttentionModule(dim, num_heads, dropout)
        self.conv = ConformerConvModule(dim, conv_kernel_size, dropout)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        # FF1
        x = x + 0.5 * self.ff1(x)
        # Self-Attention
        x = self.attn(x)
        # Convolution
        x = x + self.conv(x)
        # FF2
        x = x + 0.5 * self.ff2(x)
        # Final Layer Norm
        return self.layer_norm(x)


################################################################################
# Conformer Encoder
################################################################################
class ConformerEncoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=128, num_layers=6, num_heads=8, ff_expansion_factor=4,
                 conv_kernel_size=31, dropout=0.1):
        super().__init__()
        # 前端特征提取层(例如卷积下采样)
        self.subsampling = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.layer_norm = nn.LayerNorm(output_dim)
        self.blocks = nn.ModuleList([
            ConformerBlock(output_dim, num_heads, ff_expansion_factor, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: (B, C=2, T)
        # Subsampling
        x = self.subsampling(x)  # (B, D, T/4) approx.
        x = x.transpose(1, 2)  # (B, T', D)
        x = self.layer_norm(x)
        for block in self.blocks:
            x = block(x)
        return x  # (B, T', D)


################################################################################
# Prediction Network
################################################################################
class PredictionNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, y_in, hidden=None):
        # y_in: (B, U) target sequence without future context
        emb = self.embedding(y_in)  # (B, U, embed_dim)
        out, hidden = self.lstm(emb, hidden)  # (B, U, hidden_dim)
        return out, hidden


################################################################################
# Joint Network
################################################################################
class JointNetwork(nn.Module):
    def __init__(self, enc_dim=128, pred_dim=128, vocab_size=100):
        super().__init__()
        # Linear layers to combine encoder and predictor outputs
        self.fc = nn.Linear(enc_dim + pred_dim, vocab_size)

    def forward(self, enc_out, pred_out):
        # enc_out: (B, T, D_enc)
        # pred_out: (B, U, D_pred)
        # 需要将二者broadcast后融合，例如:
        # enc_out_expanded: (B, T, U, D_enc)
        # pred_out_expanded: (B, T, U, D_pred) after unsqueeze and broadcasting
        # 最简单的方法是使用广播加法之前先unsqueeze
        # 这里使用expand会导致内存大，实际可用更经济的写法

        # 扩维
        B, T, D_enc = enc_out.size()
        B_, U, D_pred = pred_out.size()
        # enc_out.unsqueeze(2): (B, T, 1, D_enc)
        # pred_out.unsqueeze(1): (B, 1, U, D_pred)
        enc_expanded = enc_out.unsqueeze(2)  # (B, T, 1, D_enc)
        pred_expanded = pred_out.unsqueeze(1)  # (B, 1, U, D_pred)

        # 广播拼接
        # 最终形状: (B, T, U, D_enc + D_pred)
        combined = torch.cat([enc_expanded.expand(B, T, U, D_enc),
                              pred_expanded.expand(B, T, U, D_pred)], dim=-1)
        # Linear
        logits = self.fc(combined)  # (B, T, U, vocab_size)
        return logits


################################################################################
# Conformer-RNN-T Model整合
################################################################################
class ConformerRNNTModel(nn.Module):
    def __init__(self, vocab_size, blank_idx, bos_idx, enc_dim=128, pred_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.blank_idx = blank_idx
        self.bos_idx = bos_idx

        self.encoder = ConformerEncoder(input_dim=2, output_dim=enc_dim)
        self.pred_net = PredictionNetwork(vocab_size, embed_dim=pred_dim, hidden_dim=pred_dim)
        self.joint = JointNetwork(enc_dim=enc_dim, pred_dim=pred_dim, vocab_size=vocab_size)

    def forward(self, src, tgt):
        # src: (B, 2, T_in)
        # tgt: (B, U) 不含未来信息的真实序列

        enc_out = self.encoder(src)  # (B, T', D_enc)
        pred_out, _ = self.pred_net(tgt)  # (B, U, D_pred)
        logits = self.joint(enc_out, pred_out)  # (B, T', U, vocab_size)
        return logits

    def compute_loss(self, src, tgt, tgt_len):
        # tgt: (B, U) 包括EOS但不含BOS的目标，或者根据需要在外部添加BOS
        # tgt_len: (B,) 对应每条序列的长度
        # 注意：RNNT损失需要encoder长度和target长度
        enc_out = self.encoder(src)  # (B, T', D_enc)
        pred_out, _ = self.pred_net(tgt)  # (B, U, D_pred)
        logits = self.joint(enc_out, pred_out)  # (B, T', U, vocab_size)

        # RNNT Loss需要logits, src_lengths, target_lengths, target_labels
        # 假设src_lengths和tgt_len在外部已计算好
        # logits的形状期望： (B, max_T, max_U, vocab_size)
        # 使用torchaudio的RNNT Loss
        src_len = torch.full((src.size(0),), enc_out.size(1), dtype=torch.int32, device=src.device)
        tgt_len = tgt_len.to(torch.int32)

        loss = torchaudio.functional.rnnt_loss(
            logits=logits,
            targets=tgt,
            src_lengths=src_len,
            target_lengths=tgt_len,
            blank=self.blank_idx,
            reduction='mean'
        )
        return loss

    def greedy_decode(self, src):
        # 简化的Greedy推理流程 (非流式，仅示意)
        self.eval()
        with torch.no_grad():
            enc_out = self.encoder(src)  # (B, T', D_enc)
            B, T, D = enc_out.size()
            # 初始化Prediction
            pred_token = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=src.device)
            pred_hid = None
            u = 0
            results = [[] for _ in range(B)]
            t = 0
            # Greedy: 遍历Encoder时序
            # RNN-T解码一般需要同时遍历T和U，通过blank决定如何移动
            # 简单的过程（仅示意真实实现需要更复杂的循环和条件判断）：
            for t_i in range(T):
                # 对当前frame重复尝试扩展U方向
                while True:
                    pred_out, pred_hid = self.pred_net(pred_token, pred_hid)  # (B, u+1, D_pred)
                    joint_out = self.joint(enc_out[:, t_i:t_i + 1, :], pred_out[:, -1:, :])  # (B, 1, 1, vocab_size)
                    joint_out = joint_out.squeeze(1).squeeze(1)  # (B, vocab_size)
                    next_char = torch.argmax(joint_out, dim=-1)
                    if (next_char == self.blank_idx).all():
                        # 全部预测blank，移动encoder时间维
                        break
                    else:
                        # 对于非blank的token进行添加并继续尝试下一个U
                        for b_i in range(B):
                            if next_char[b_i] != self.blank_idx:
                                results[b_i].append(next_char[b_i].item())
                        pred_token = torch.cat([pred_token, next_char.unsqueeze(-1)], dim=1)
        return results

### 总结
#
# 上述代码展示了如何构建Conformer - RNN - T模型来完成IQ流到码序列的映射任务。与传统的生成式Transformer不同：
#
# - ** RNN - T无需显式对齐 **：通过引入blank标签和RNN - T
# loss实现自动对齐。
# - ** Conformer作为编码器 **：更适合捕捉时频特征和上下文信息，有助于从IQ数据中提取更鲁棒的特征。
# - ** Prediction
# Network和Joint
# Network **：为RNN - T特有的结构，使得模型能够在无对齐条件下端到端训练。
#
# 实际应用中还需对输入特征（如对IQ数据做预处理、归一化、同步）、模型超参数（layer数量、维度、head数量）以及优化策略（学习率调度、数据增强）进行细致调试。
