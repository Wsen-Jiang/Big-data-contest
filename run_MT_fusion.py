import os
import torch
import csv
from models.ResBlock_Classifier import ResBlock_Classifier
from models.CNN_LSTM_Classifier import CNN_LSTM_Classifier
from models.Triplet_Classifier import Triplet_Classifier
from utils.dataset import load_data_from_directories
from sklearn.model_selection import train_test_split
import itertools


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

    # 模型路径
    MT_model_paths = [
        "/mnt/data/JWS/Big-data-contest/log/models/ModulationType/ResBlock_Classifier/MT_ResBlock_68.443_model.pth",
        "/mnt/data/JWS/Big-data-contest/log/models/ModulationType/Triplet_Classifier/MT_70.04_best_model.pth",
        "/mnt/data/JWS/Big-data-contest/log/models/ModulationType/CNN_LSTM_Classifier/MT_CNN_LSTM_70.59_best_model.pth"
    ]

    # 模型类列表
    model_classes = [ResBlock_Classifier, Triplet_Classifier, CNN_LSTM_Classifier]  # 可根据实际情况调整为不同类型的模型类
    models = []

    # 加载所有模型
    for MT_model_path, model_class in zip(MT_model_paths, model_classes):
        model = model_class().to(device)  # 根据类型实例化模型

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
            model_keys = set(model.state_dict().keys())
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
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
            print(f"[成功] 模型已成功加载: {MT_model_path}")

        except RuntimeError as e:
            print(f"[错误] 模型加载失败: {e}")
            return False

    # 权重范围 [0.4, 0.6] 步长 0.05
    weight_range_1 = [i * 0.05 + 0.4 for i in range(5)]  # [0.4, 0.45, 0.5, 0.55, 0.6]

    # 权重范围 [0.15, 0.35] 步长 0.05
    weight_range_2 = [i * 0.05 + 0.15 for i in range(5)]  # [0.15, 0.2, 0.25, 0.3, 0.35]

    # 权重范围 [0.15, 0.35] 步长 0.05
    weight_range_3 = [i * 0.05 + 0.15 for i in range(5)]  # [0.15, 0.2, 0.25, 0.3, 0.35]

    best_accuracy = 0
    best_weights = None

    # 创建 CSV 文件
    result_file = "model_weights_accuracy.csv"
    with open(result_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Weight 1", "Weight 2", "Weight 3", "Accuracy"])  # 写入表头

        # 遍历所有权重组合
        for weights in itertools.product(weight_range_1, weight_range_2, weight_range_3):
            if sum(weights) != 1:  # 权重和必须为1
                continue

            # 开始预测
            correct = 0
            total = len(seq_val)

            with torch.no_grad():
                for i, (seq, val) in enumerate(zip(seq_val, y_val)):
                    seq = seq.to(device)  # 将 seq 移动到 device
                    val = val.clone().detach().to(device)
                    seq = seq.unsqueeze(0)  # 增加一个 batch_size 维度

                    # 预处理: permute 以匹配模型输入
                    seq = seq.permute(0, 2, 1)

                    # 初始化融合概率
                    fused_probs = torch.zeros(1, 10).to(device)  # 假设有10个类别

                    # 对每个模型的预测结果加权求和
                    for model, weight in zip(models, weights):
                        outputs = model(seq)  # 获取 logits
                        probs = torch.nn.functional.softmax(outputs, dim=1)  # 转换为概率
                        fused_probs += weight * probs  # 加权求和

                    # 归一化融合概率（确保权重之和为1）
                    fused_probs = fused_probs / sum(weights)

                    # 获取预测的类别
                    predicted = torch.argmax(fused_probs, dim=1) + 1

                    # 计算正确预测数量
                    correct += (predicted == (val + 1)).sum().item()  # 根据原代码，标签可能是从0开始

            accuracy = correct / total
            print(f"当前权重组合 {weights} 的准确率为：{accuracy:.4f}")

            # 将结果写入CSV文件
            writer.writerow([*weights, accuracy])

            # 更新最优结果
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = weights

    print(f"最优权重组合为：{best_weights}，准确率为：{best_accuracy:.4f}")


if __name__ == "__main__":
    main()
