import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse
import importlib
from dataset import load_data_from_directories, WaveformDataset, CollateFunction
from utils import show_plot
from Criterion.SW_RelativeErrorLoss import RelativeErrorLoss
from Criterion.CosineSimilarityLoss import CosineSimilarityLoss
from sklearn.metrics import mean_squared_error
import numpy as np

# 计算码元宽度得分
def calculate_score(relative_error):
    if relative_error <= 0.05:
        return 100
    elif relative_error >= 0.20:
        return 0
    else:
        # 在 5% 和 20% 之间线性下降
        return 100 - ((relative_error - 0.05) / (0.20 - 0.05)) * 100

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, task):

    # 设置最佳指标：码元宽度回归任务以最小化损失为目标，调制类型分类任务以最大化准确率为目标，码序列相似度任务以最大化余弦相似度为目标
    best_metric = float('inf') if task == "SW" else 0

    history_train_loss = []
    history_valid_loss = []

    if task == "MT":
        end_path = "ModulationType"
    elif task == "SW":
        end_path = "SymbolWidth"
    elif task == "CQ":
        end_path = "CodeSequence"
    else:
        raise ValueError(f"无效的 task 参数: {task}")

    model_dir = f'log/models/{end_path}/{model.__class__.__name__}'
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0

        for batch_X, seq_lengths, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            seq_lengths = seq_lengths.to(device)

            # 调整输入数据的形状
            batch_X = batch_X.permute(0, 2, 1)  # [batch_size, channels, seq_len]

            # print("Input shape:", batch_X.shape)

            optimizer.zero_grad()
            outputs = model(batch_X)

            if task == "SW":
                loss = criterion(outputs, batch_y.view(-1, 1))  # 回归任务
            else:
                loss = criterion(outputs, batch_y)  # 分类任务

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)

        history_train_loss.append(train_loss)

        # 验证阶段
        model.eval()
        sum_val_loss = 0.0
        if task in ["MT", "CQ"]:
            correct = 0
            total = 0
        else:
            all_preds = []
            all_targets = []

        with torch.no_grad():
            for batch_X, seq_lengths, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                seq_lengths = seq_lengths.to(device)

                batch_X = batch_X.permute(0, 2, 1)  # [batch_size, channels, seq_len]

                outputs = model(batch_X)

                if task == "SW":
                    loss = criterion(outputs, batch_y.view(-1, 1))
                    sum_val_loss += loss.item()  # 累加总损失
                    all_preds.extend(outputs.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
                else:
                    loss = criterion(outputs, batch_y)
                    sum_val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

        if task == "SW":
            avg_val_loss = sum_val_loss / len(val_loader)  # 计算平均损失
            mse = mean_squared_error(all_targets, all_preds)
            rmse = mean_squared_error(all_targets, all_preds, squared=False)

            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')
            if avg_val_loss < best_metric:
                best_metric = avg_val_loss
                # 保存最佳模型
                torch.save(model.state_dict(), os.path.join(model_dir, f'best_model.pth'))
        else:
            avg_val_loss = sum_val_loss / len(val_loader)  # 对于分类任务，可以保持原有方式
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
            if accuracy > best_metric:
                best_metric = accuracy
                # 保存最佳模型
                torch.save(model.state_dict(), os.path.join(model_dir, f'best_model.pth'))

        history_valid_loss.append(avg_val_loss)

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(model_dir,'final_model.pth'))

    # loss图保存路径
    save_path = f'log/save_loss/{end_path}/{model.__class__.__name__}'
    os.makedirs(save_path, exist_ok=True)
    show_plot(history_train_loss, history_valid_loss,
              f"{save_path}/{model.__class__.__name__}_{args.lr}_{args.batch_size}_{best_metric}.png")


def test(model, val_loader, criterion, device, model_path, task):
    # 加载已保存的模型权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型已成功加载: {model_path}")
    else:
        print(f"模型文件不存在: {model_path}")
        return

    model.eval()

    val_loss = 0.0
    if task in ["MT", "CQ"]:
        correct = 0
        total = 0
    else:
        all_preds = []
        all_targets = []

    with torch.no_grad():
        for batch_X, seq_lengths, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            seq_lengths = seq_lengths.to(device)

            batch_X = batch_X.permute(0, 2, 1)  # [batch_size, channels, seq_len]

            outputs = model(batch_X)

            if task == "SW":
                loss = criterion(outputs, batch_y.view(-1, 1))
                val_loss += loss.item()  # 累加总损失
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
            else:
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

    if task in ["MT", "CQ"]:
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        print(f'验证集上的损失: {avg_val_loss:.4f}, 准确率: {accuracy:.3f}%')
    elif task == "SW":
        avg_val_loss = val_loss / len(val_loader)
        mse = mean_squared_error(all_targets, all_preds)
        rmse = mean_squared_error(all_targets, all_preds, squared=False)
        #计算相对误差
        all_preds = np.array(all_preds).squeeze() #将预测结果降为一维，防止触发广播机制
        # 对 all_preds 的值进行裁剪，限定范围在 0.2 到 1 之间
        all_preds = np.clip(all_preds, 0.2, 1.0)

        all_targets = np.array(all_targets)
        relative_error = np.abs(all_preds - all_targets) / all_targets # all_targets为正数

        mean_relative_error = np.mean(relative_error)

        # 对每个样本的 relative_error 计算得分
        scores = np.array([calculate_score(re) for re in relative_error])
        # 计算总得分 (即所有样本得分的平均值)
        total_score = np.mean(scores)

        print(f'验证集上的损失: {avg_val_loss:.4f}, MeanRelativeError：{mean_relative_error:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')
        print(f"总得分: {total_score:.2f}")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("Train or test the model")
    arg_parser.add_argument("--mode", type=str, default="train", help="train or test")
    arg_parser.add_argument("--task", type=str, default="MT",
                            help="MT(ModulationType)、SW(SymbolWidth)、CQ(CodeSequence)")
    arg_parser.add_argument("--network", type=str, default="CNNClassifier",
                            help="选择网络 (例如 CNNClassifier, ResNet)")
    arg_parser.add_argument("--lr", type=float, default=0.0005, help="学习率")
    arg_parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    arg_parser.add_argument("--batch_size", type=int, default=512, help="批次大小")
    arg_parser.add_argument("--model_path", type=str, default="",
                            help="模型文件路径，用于测试模式")
    args = arg_parser.parse_args()

    # 指定根目录
    root_dir = 'Dataset'
    data_dirs = ['1', '2', '3', '4']

    # 读取数据
    sequences, labels = load_data_from_directories(root_dir, data_dirs, args.task)

    # 划分训练集和验证集
    if args.task == "MT":
        stratify = labels
    else:
        stratify = None

    seq_train, seq_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=stratify
    )
    collate_fn = CollateFunction(args.task)
    # 创建数据集和数据加载器
    if args.mode == 'train':
        train_dataset = WaveformDataset(seq_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataset = WaveformDataset(seq_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    elif args.mode == 'test':
        val_dataset = WaveformDataset(seq_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        raise ValueError("无效的模式，请选择 'train' 或 'test'")

    # 动态导入或选择网络模型
    try:
        module = importlib.import_module(f"models.{args.network}")
        model_class = getattr(module, args.network)
        model = model_class()
    except (ImportError, AttributeError):
        raise ValueError(f"指定的网络 {args.network} 无效或不存在")

    # 定义损失函数和优化器
    if args.task == "MT":
        criterion = nn.CrossEntropyLoss()
    elif args.task == "SW":
        criterion = RelativeErrorLoss()
    elif args.task == "CQ":
        criterion = CosineSimilarityLoss()

    else:
        raise ValueError(f"无效的 task 参数: {args.task}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if args.mode == 'train':
        # 训练模型
        train(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, args.task)
    elif args.mode == 'test':
        if not args.model_path:
            print("测试模式需要指定模型文件路径，请使用 --model_path 参数。")
            exit(1)
        test(model, val_loader, criterion, device, args.model_path, args.task)

"""测试指令
python main.py --mode test --task SW --network CNNRegressor --model_path log/models/SymbolWidth/CNNRegressor/0.1495_60.81_best_model.pth"""
