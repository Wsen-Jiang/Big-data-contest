import os
import torch
from models.ResBlock_Regressor import ResBlock_Regressor
from utils.dataset import load_data_from_directories
from sklearn.model_selection import train_test_split
import numpy as np

def calculate_score(relative_error):
    if relative_error <= 0.05:
        return 100
    elif relative_error >= 0.20:
        return 0
    else:
        # 在 5% 和 20% 之间线性下降
        return 100 - ((relative_error - 0.05) / (0.20 - 0.05)) * 100

def main():
    """
    主函数，用于执行脚本的主要逻辑。

    参数：
        to_pred_dir: 测试集文件夹上层目录路径，不可更改!
        result_save_path: 预测结果文件保存路径，官方已指定为csv格式，不可更改!
    """
    root_dir = 'train_data'
    data_dirs = ['8APSK', '8PSK', '8QAM', '16APSK', '16QAM', '32APSK', '32QAM', 'BPSK', 'MSK', 'QPSK']

    # 读取数据
    sequences, labels = load_data_from_directories(root_dir, data_dirs, "SW")

    # 划分训练集和验证集
    seq_train, seq_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    SW_model_path = r"/mnt/data/JWS/Big-data-contest/log/models/SymbolWidth/ResBlock_Regressor/SW_97.89_best_model.pth"
    SW_model = ResBlock_Regressor()
    SW_model.to(device)

    # 加载模型
    if not os.path.exists(SW_model_path):
        print(f"[错误] 模型文件不存在: {SW_model_path}")
        return False

    try:
        # 加载权重
        checkpoint = torch.load(SW_model_path, map_location=device, weights_only=True)

        # 检查权重是否是字典格式
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 检查是否需要添加或移除 `module.` 前缀
        model_keys = set(SW_model.state_dict().keys())
        state_keys = set(state_dict.keys())

        if all(key.startswith("module.") for key in state_keys) and not any(
                key.startswith("module.") for key in model_keys):
            # 权重有 `module.` 前缀，但模型没有
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        elif not any(key.startswith("module.") for key in state_keys) and all(
                key.startswith("module.") for key in model_keys):
            # 模型有 `module.` 前缀，但权重没有
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}

        # 加载状态字典
        SW_model.load_state_dict(state_dict)
        print(f"[成功] 模型已成功加载: {SW_model_path}")

    except RuntimeError as e:
        print(f"[错误] 模型加载失败: {e}")
        return False

    # 开始预测
    SW_model.eval()
    all_score = 0
    with torch.no_grad():
        for seq, val in zip(seq_val, y_val):
            seq = seq.clone().detach().unsqueeze(0).permute(0, 2, 1).to(device)  # 数据移动到设备
            val = val.clone().detach().to(device)

            # 预测码元宽度
            predict_SW = SW_model(seq).item()
            predict_SW = round(predict_SW/0.05)*0.05
            label = round(val.item(), 2)
            score_error = np.abs(predict_SW - label)
            score = calculate_score(score_error)
            # print(f"模型预测宽度：{predict_SW:.2f}            真实标签：{val}")
            all_score += score

    accuracy = (all_score / len(seq_val))

    print("预测得分为：{}".format(accuracy))


if __name__ == "__main__":
    main()
