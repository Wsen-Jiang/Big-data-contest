import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, predicted_sequences, target_sequences, label_lengths):
        batch_size, seq_length = predicted_sequences.size()

        # 基于 label_lengths 创建 mask
        mask = torch.arange(seq_length).unsqueeze(0).expand(batch_size, seq_length).to(label_lengths.device)
        mask = mask < label_lengths.unsqueeze(1)

        # 将 predicted_sequences 和 target_sequences 转为浮点数类型，确保余弦相似度计算时类型匹配
        predicted_sequences = predicted_sequences.float()
        target_sequences = target_sequences.float()

        # 应用 mask：保持有效位置，将无效（填充）位置设置为 0(0不会对余弦相似度的计算产生影响)
        masked_predicted = torch.where(mask, predicted_sequences, torch.zeros_like(predicted_sequences))
        # 沿着嵌入维度计算余弦相似度
        cos_sim = F.cosine_similarity(masked_predicted, target_sequences, dim=-1)  # 形状: (batch_size, seq_length)

        loss = 1 - cos_sim.mean()  # 计算整个批次的平均余弦相似度损失

        return loss
