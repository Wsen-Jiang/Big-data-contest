import os
import torch
from models.CNN_Regressor_LSTM import CNN_Regressor_LSTM
from dataset import load_data_from_directories
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SW_model_path = r'/mnt/data/LWP/Signal-Test/log/models/SymbolWidth/CNN_Regressor_LSTM/0.0486_88.74_best_model.pth'
    SW_model = CNN_Regressor_LSTM()

    # 加载模型
    if os.path.exists(SW_model_path):
        SW_model.load_state_dict(torch.load(SW_model_path, map_location=device))
        print(f"模型已成功加载: {SW_model_path}")
    else:
        print(f"模型文件不存在: {SW_model_path}")
        return

    # 开始预测
    SW_model.eval()
    all_score = 0
    with torch.no_grad():
        for idx, val in enumerate(seq_val):
            sequence = val.unsqueeze(0)  # 增加一个 batch_size 维度，变为 [1, 1727, 2]
            sequence = sequence.permute(0, 2, 1)  # 转置，变为 [1, 2, 1727]

            # 预测码元宽度
            predict_SW = SW_model(sequence).item()
            score_error = np.abs(predict_SW - y_val[idx])
            score = calculate_score(score_error)
            print(f"模型预测宽度：{predict_SW:.2f}            真实标签：{y_val[idx]}")
            all_score += score

    accuracy = (all_score / len(seq_val))

    print("预测得分为：{}".format(accuracy))


if __name__ == "__main__":
    main()
