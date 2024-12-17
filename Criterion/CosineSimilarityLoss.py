import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineSimilarityLoss(nn.Module):
    def __init__(self, vocab_size, pad_idx=0):
        super(CosineSimilarityLoss, self).__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

    def forward(self, logits, targets):
        """
        logits: (batch_size, seq_length, vocab_size) - 模型输出的 logits
        targets: (batch_size, seq_length) - 真实的标签索引
        """
        # 计算概率分布
        probs = F.softmax(logits, dim=-1)  # (batch_size, seq_length, vocab_size)

        # 将 targets 转换为 one-hot 编码
        targets_one_hot = F.one_hot(targets, num_classes=self.vocab_size).float()  # (batch_size, seq_length, vocab_size)

        # 计算余弦相似度
        # 防止除以零
        epsilon = 1e-8
        cosine_sim = F.cosine_similarity(probs, targets_one_hot, dim=-1)  # (batch_size, seq_length)

        # 创建掩码，忽略 <PAD> 的位置
        mask = (targets != self.pad_idx).float()  # (batch_size, seq_length)

        # 计算平均余弦相似度
        cosine_sim = cosine_sim * mask  # (batch_size, seq_length)
        mean_cosine_sim = cosine_sim.sum() / (mask.sum() + epsilon)

        # 损失为 1 - 余弦相似度，使得相似度越高，损失越低
        loss = 1 - mean_cosine_sim
        return loss
