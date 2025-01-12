import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.CNN_LSTM_Classifier import CNN_LSTM_Classifier
from models.ResBlock_Classifier import ResBlock_Classifier
from utils.dataset import load_data_from_directories
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid


def main():
    """
    主函数，用于加载两个模型，对验证集进行加权融合预测，并计算准确率。
    """
    root_dir = 'train_data'
    data_dirs = ['8APSK', '8PSK', '8QAM', '16APSK', '16QAM', '32APSK', '32QAM', 'BPSK', 'MSK', 'QPSK']

    # 读取数据
    sequences, labels = load_data_from_directories(root_dir, data_dirs, "MT")

    # 划分训练集和验证集
    seq_train, seq_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # 模型路径
    MT_model_path = r"/mnt/data/JWS/Big-data-contest/log/models/ModulationType/CNN_LSTM_Classifier/MT_best_model.pth"
    RB_model_path = r"/mnt/data/JWS/Big-data-contest/log/models/SymbolWidth/best_model_64.72.pth"

    # 初始化模型
    MT_model = CNN_LSTM_Classifier()
    RB_model = ResBlock_Classifier()
    MT_model.to(device)
    RB_model.to(device)

    # 加载第一个模型的权重
    if os.path.exists(MT_model_path):
        checkpoint = torch.load(MT_model_path, map_location=device, weights_only=True)
        # 如果 checkpoint 是一个包含 'state_dict' 的字典
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        # 去掉 "module." 前缀（如果存在）
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        MT_model.load_state_dict(state_dict)
        print(f"模型已成功加载: {MT_model_path}")
    else:
        print(f"模型文件不存在: {MT_model_path}")
        return

    # 加载第二个模型的权重
    if os.path.exists(RB_model_path):
        checkpoint = torch.load(RB_model_path, map_location=device, weights_only=True)
        # 如果 checkpoint 是一个包含 'state_dict' 的字典
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        # 去掉 "module." 前缀（如果存在）
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        RB_model.load_state_dict(state_dict)
        print(f"模型已成功加载: {RB_model_path}")
    else:
        print(f"模型文件不存在: {RB_model_path}")
        return

    # 设置模型为评估模式
    MT_model.eval()
    RB_model.eval()

    # 获取两个模型的预测概率和真实标签
    all_probs1 = []  # CNN_LSTM_Classifier 的概率
    all_probs2 = []  # ResBlock_Classifier 的概率
    all_targets = []

    with torch.no_grad():
        for data, target in zip(seq_val, y_val):
            data = data.clone().detach().unsqueeze(0).permute(0, 2, 1).to(device)
            target = target.to(device)

            # 第一个模型的输出
            output1 = MT_model(data)
            probs1 = nn.functional.softmax(output1, dim=1)
            all_probs1.append(probs1.cpu().numpy())

            # 第二个模型的输出
            output2 = RB_model(data)
            probs2 = nn.functional.softmax(output2, dim=1)
            all_probs2.append(probs2.cpu().numpy())

            all_targets.extend(target.cpu().numpy().flatten())
    # 将 all_probs1 和 all_probs2 转换为 NumPy 数组
    all_probs1 = np.array(all_probs1).squeeze(axis=1)  # 去除第二个维度
    all_probs2 = np.array(all_probs2).squeeze(axis=1)  # 去除第二个维度

    # 定义权重搜索空间（w1 + w2 = 1）
    param_grid = {'w1': np.linspace(0, 1, 1001)}  # w1 从0到1，步长为0.1
    best_acc = 0
    best_w1 = 0.5

    # 网格搜索找到最佳权重
    for params in ParameterGrid(param_grid):
        w1 = params['w1']
        w2 = 1 - w1
        # 加权融合概率
        ensemble_probs = w1 * all_probs1 + w2 * all_probs2
        # 预测
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        # 计算准确率
        acc = accuracy_score(all_targets, ensemble_preds)
        print(f"w1: {w1:.1f}, w2: {w2:.1f}, Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_w1 = w1

    best_w2 = 1 - best_w1
    print(f"\n最佳权重组合: model1 (w1) = {best_w1:.2f}, model2 (w2) = {best_w2:.2f}, 准确率 = {best_acc:.4f}")

    # 使用最佳权重进行最终预测
    ensemble_probs_best = best_w1 * all_probs1 + best_w2 * all_probs2
    ensemble_preds_best = np.argmax(ensemble_probs_best, axis=1)
    acc_best = accuracy_score(all_targets, ensemble_preds_best)
    print(f"加权融合后的最终准确率: {acc_best:.4f}")

    # 计算每个类别的准确率
    ls_all_num = [0 for _ in range(10)]
    ls_correct_num = [0 for _ in range(10)]
    ls_name = ['BPSK', 'QPSK', '8PSK', 'MSK', '8QAM', '16QAM', '32QAM', '8APSK', '16APSK', '32APSK']

    for pred, true in zip(ensemble_preds_best, all_targets):
        ls_all_num[true] += 1
        if pred == true:
            ls_correct_num[true] += 1

    accuracy = acc_best
    print(f"总体准确率为：{accuracy:.4f}")
    for i in range(10):
        class_acc = ls_correct_num[i] / ls_all_num[i] if ls_all_num[i] > 0 else 0
        print(f"{ls_name[i]} 调制类型的准确率为：{class_acc:.4f}")



if __name__ == "__main__":
    main()