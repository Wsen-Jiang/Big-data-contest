import torch
import torch.nn.functional as F

def cosine_similarity_seq(pred, target, lengths, padding_value=0):
    # pred: (batch_size, max_len) 预测码序列
    # target: (batch_size, max_len) 真实标签序列
    # lengths: (batch_size,) 各个样本的实际长度
    device = pred.device
    batch_size, max_target_len = target.size()
    _,max_pred_len = pred.size()


    if max_target_len > max_pred_len:
        # 填充预测序列
        pad = torch.full((batch_size, max_target_len - max_pred_len), padding_value, dtype=torch.long, device=device)
        pred = torch.cat([pred, pad], dim=1)
    else:
        # 截断预测序列
        pred = pred[:, :max_target_len]

    # 构造掩码
    mask = (torch.arange(max_target_len, device=device).unsqueeze(0) < lengths.unsqueeze(1))  # (batch_size, max_len)

    pred = (pred * mask).to(torch.float32) # (batch_size, max_len)
    target = (target * mask).to(torch.float32)# (batch_size, max_len)
    # 计算余弦相似度
    cosine_sims = F.cosine_similarity(pred, target, dim=1)  # (batch_size,)

    # 返回平均相似度
    return cosine_sims.mean().item()
