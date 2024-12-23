import matplotlib.pyplot as plt
import torch
import random
import os
import numpy as np
import torch.backends.cudnn as cudnn

def show_plot(train_losses_history, valid_losses_history, save_pic):
    # 确保所有张量都在 CPU 上并转换为 NumPy 数组
    train_losses_history = [loss.cpu().detach().numpy() if isinstance(loss, torch.Tensor) else loss for loss in train_losses_history]
    valid_losses_history = [loss.cpu().detach().numpy() if isinstance(loss, torch.Tensor) else loss for loss in valid_losses_history]

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(train_losses_history, label='Training Loss', color='blue')
    plt.plot(valid_losses_history, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_pic)
    plt.show()


def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True