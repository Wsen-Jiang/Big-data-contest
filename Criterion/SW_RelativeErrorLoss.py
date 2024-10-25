import torch
import torch.nn as nn

class RelativeErrorLoss(nn.Module):
    def __init__(self):
        super(RelativeErrorLoss, self).__init__()

    def forward(self, predictions, targets):
        # 避免除零错误，在分母中添加一个小常数 epsilon
        epsilon = 1e-6
        relative_error = torch.abs((predictions - targets) / (torch.abs(targets) + epsilon))
        loss = torch.mean(relative_error)  # 对相对误差取均值
        return loss
