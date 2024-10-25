import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityLoss(nn.Module):
    def __init__(self, decoder):
        super(CosineSimilarityLoss, self).__init__()
        self.decoder = decoder  # 需要传入解码器以便获取嵌入

    def forward(self, outputs, labels, label_lengths):
        # 截断输出以匹配真实标签长度
        batch_size, max_len, output_dim = outputs.size()

        total_loss = 0.0

        for i in range(batch_size):
            current_output = outputs[i, :label_lengths[i], :]  # (label_lengths[i], output_dim)
            current_label = labels[i, :label_lengths[i]]  # (label_lengths[i])

            # 使用解码器的嵌入层将真实标签转化为嵌入
            one_hot_labels = self.decoder.embedding(current_label)  # (label_lengths[i], emb_dim)

            # 计算余弦相似度
            similarity = F.cosine_similarity(current_output, one_hot_labels, dim=1)

            # 将相似度转为损失（1 - similarity）
            loss = 1 - similarity.mean()  # 平均损失
            total_loss += loss

        return total_loss / batch_size
