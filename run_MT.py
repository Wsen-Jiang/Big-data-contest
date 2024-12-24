import os
import torch
from models.ResBlock_Classifier import ResBlock_Classifier
from utils.dataset import load_data_from_directories
from sklearn.model_selection import train_test_split
import numpy as np

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
    MT_model_path = r"/mnt/data/JWS/Big-data-contest/log/models/ModulationType/ResBlock_Classifier/67.14_best_model.pth"
    MT_model = ResBlock_Classifier()
    MT_model.to(device)

   # 加载模型
    if os.path.exists(MT_model_path):
        MT_model.load_state_dict(torch.load(MT_model_path, map_location=device, weights_only=True))
        print(f"模型已成功加载: {MT_model_path}")
    else:
        print(f"模型文件不存在: {MT_model_path}")
        return

    # 开始预测
    MT_model.eval()
    correct = 0
    list = []
    with torch.no_grad():
        for seq, val in zip(seq_val, y_val):

            seq = seq.to(device)  # 将seq移动到device
            val = val.clone().detach().to(device)
            seq = seq.unsqueeze(0)  # 增加一个 batch_size 维度

            # 单个验证样本的模型输出
            seq = seq.permute(0, 2, 1)
            outputs = MT_model(seq)

            logits = torch.nn.functional.softmax(outputs, dim=1)
            # 选出最高的概率
            max_score = torch.max(logits).item()

            modulation_type = torch.argmax(outputs, dim=1) + 1  # 调制类型预测
            if (modulation_type == (val+1)).item():
                correct += 1
                list.append(max_score)

            # print(f'模型预测调制类型：{modulation_type.item()}, 实际真实label:{real_label}')

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