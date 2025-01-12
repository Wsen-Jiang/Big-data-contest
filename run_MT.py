import os
import torch
from models.ResBlock_Classifier import ResBlock_Classifier
from utils.dataset import load_data_from_directories
from sklearn.model_selection import train_test_split
import numpy as np
from models.CNN_LSTM_Classifier import CNN_LSTM_Classifier
from models.Triplet_Classifier import Triplet_Classifier

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
    sequences, labels = load_data_from_directories(root_dir, data_dirs, "MT")

    # 划分训练集和验证集
    seq_train, seq_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    MT_model_path = r"/mnt/data/JWS/Big-data-contest/log/models/ModulationType/CNN_LSTM_Classifier/MT_CNN_LSTM_70.59_best_model.pth"
    MT_model = CNN_LSTM_Classifier()
    MT_model.to(device)


    if not os.path.exists(MT_model_path):
        print(f"[错误] 模型文件不存在: {MT_model_path}")
        return False

    try:
        # 加载权重
        checkpoint = torch.load(MT_model_path, map_location=device, weights_only=True)

        # 检查权重是否是字典格式
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 检查是否需要添加或移除 `module.` 前缀
        model_keys = set(MT_model.state_dict().keys())
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
        MT_model.load_state_dict(state_dict)
        print(f"[成功] 模型已成功加载: {MT_model_path}")

    except RuntimeError as e:
        print(f"[错误] 模型加载失败: {e}")
        return False

    # 开始预测
    MT_model.eval()
    correct = 0
    list = []
    with torch.no_grad():
        for seq, val in zip(seq_val, y_val):

            val = val.clone().detach().to(device)
            seq = seq.clone().detach().unsqueeze(0).permute(0, 2, 1).to(device)  # 数据移动到设备

            outputs = MT_model(seq)

            logits = torch.nn.functional.softmax(outputs, dim=1)
            # 选出最高的概率
            max_score = torch.max(logits).item()

            modulation_type = torch.argmax(outputs, dim=1) + 1  # 调制类型预测
            if (modulation_type == (val+1)).item():
                correct += 1
                list.append(max_score)

    accuracy = correct / len(seq_val)
    print(f"准确率为：{accuracy:.4f}")
    return list

if __name__ == "__main__":
    list = main()
    # 绘制list散点图
    import matplotlib.pyplot as plt

    # 定义直方图区间
    bins = 10  # 分成 10 个区间
    range_min, range_max = 0, 1  # 数据范围 [0, 1]

    # 计算直方图数据
    hist, bin_edges = np.histogram(list, bins=bins, range=(range_min, range_max))

    # 输出每个区间的统计数量
    print("直方图统计结果：")
    for i in range(len(hist)):
        print(f"区间 {bin_edges[i]:.2f} - {bin_edges[i + 1]:.2f} 的数量: {hist[i]}")

    # 绘制直方图
    plt.hist(list, bins=bins, range=(range_min, range_max), edgecolor='black')
    plt.xlabel('Prediction Score Range')  # 横轴为分数范围
    plt.ylabel('Frequency')  # 纵轴为频率
    plt.title('Histogram of Prediction Scores')  # 标题
    plt.grid(axis='y')  # 添加网格线
    plt.show()