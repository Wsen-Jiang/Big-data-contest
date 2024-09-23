import torch
import torch.nn as nn
import torch.nn.functional as F
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, output, target, lengths):
        """
        计算输出序列和目标序列之间的平均余弦相似度的损失。

        参数：
            output (torch.Tensor): 模型的输出，形状为 [batch_size, seq_len, embedding_dim]。
            target (torch.Tensor): 目标嵌入，形状为 [batch_size, seq_len, embedding_dim]。
            lengths (torch.Tensor): 每个序列的实际长度。

        返回：
            torch.Tensor: 计算得到的损失。
        """
        batch_size = output.size(0)
        loss = 0.0

        for i in range(batch_size):
            valid_length = lengths[i]
            output_seq = output[i, :valid_length, :]  # [valid_length, embedding_dim]
            target_seq = target[i, :valid_length, :]  # [valid_length, embedding_dim]

            # 计算两个序列之间的平均余弦相似度
            cos_sim = F.cosine_similarity(output_seq, target_seq, dim=-1).mean()

            # 损失为 1 - 平均余弦相似度
            loss += (1.0 - cos_sim)

        # 对批次取平均
        loss /= batch_size
        return loss