import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse
import importlib
from utils.dataset import load_data_from_directories, WaveformDataset, CollateFunction
from utils.utils import show_plot
from Criterion.SW_RelativeErrorLoss import RelativeErrorLoss
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

# Gradient Clipping (to prevent gradient explosion)
max_norm = 5.0

def train(model, train_loader, seq_val,y_val, criterion, optimizer, device, num_epochs, task):

    # 设置最佳指标：码元宽度回归任务以最小化损失为目标，调制类型分类任务以最大化准确率为目标，码序列相似度任务以最大化余弦相似度为目标
    best_metric =0

    history_train_loss = []
    history_valid_loss = []

    if task == "MT":
        end_path = "ModulationType"
    elif task == "SW":
        end_path = "SymbolWidth"
    else:
        raise ValueError(f"无效的 task 参数: {task}")

    model_dir = f'log/models/{end_path}/{model.__class__.__name__}'
    os.makedirs(model_dir, exist_ok=True)
    # record val testing log after each epoch training
    f = open(os.path.join(model_dir, 'training.log'), 'w')

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
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.zero_grad()

            outputs = model(batch_X)

            if task == "SW":
                loss = criterion(outputs, batch_y.view(-1, 1))  # 回归任务
            else:
                loss = criterion(outputs, batch_y)  # 分类任务

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        # Step the scheduler
        # scheduler.step()

        train_loss /= len(train_loader)
        history_train_loss.append(train_loss)

        # 验证阶段
        model.eval()
        sum_val_loss = 0.0
        if task == "MT":
            correct = 0
            total = 0
        else:
            mean_score = 0
        with torch.no_grad():
            for seq, val in zip(seq_val, y_val):
                seq = seq.to(device)  # 将seq移动到device
                val = torch.tensor(val, device=device)
                seq = seq.unsqueeze(0)  # 增加一个 batch_size 维度，变为 [1, 1727, 2]
                seq = seq.permute(0, 2, 1)  # [batch_size, channels, seq_len]
                # 单个验证样本的模型输出
                outputs = model(seq)

                if task == "SW":
                    predict_SW = outputs.item()
                    score_error = np.abs(predict_SW - val.item())
                    score = calculate_score(score_error)
                    # print(f"当前样本的得分是：{score:.2f}")
                    mean_score += score
                else:
                    modulation_type = torch.argmax(outputs, dim=1) + 1  # 调制类型预测
                    correct += (modulation_type == (val+1))
        if task == "SW":
            mean_score /= len(seq_val)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Mean score: {mean_score:.2f}%')
            f.write(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy: {mean_score:.2f}%' + '\n')
            if mean_score > best_metric:
                best_metric = mean_score
                # 保存最佳模型
                torch.save(model.state_dict(), os.path.join(model_dir, f'best_model.pth'))
        else:

            accuracy = 100 * correct / len(seq_val)
            accuracy = accuracy.item()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy:.2f}%')
            f.write(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy:.2f}%' +'\n')
            if accuracy > best_metric:
                best_metric = accuracy
                # 保存最佳模型
                torch.save(model.state_dict(), os.path.join(model_dir, f'best_model.pth'))

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(model_dir,'final_model.pth'))
    f.close()


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
    if task == "MT":
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

    if task == "MT":
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
    arg_parser.add_argument("--task", type=str, default="SW",
                            help="MT(ModulationType)、SW(SymbolWidth)")
    arg_parser.add_argument("--network", type=str, default="CNN_Regressor_LSTM",
                            help="选择网络 (例如 CNNClassifier, ResNet, CNN_LSTM_Classifier)")
    arg_parser.add_argument("--lr", type=float, default=0.005, help="学习率")
    arg_parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    arg_parser.add_argument("--batch_size", type=int, default=1024, help="批次大小")
    arg_parser.add_argument("--model_path", type=str, default="",
                            help="模型文件路径，用于测试模式")
    args = arg_parser.parse_args()

    # 指定根目录
    root_dir = '/mnt/data/LXP/data/train_data'
    data_dirs = ['8APSK', '8PSK', '8QAM', '16APSK','16QAM','32APSK','32QAM','BPSK','MSK','QPSK']
    # data_dirs = ['8APSK']

    # 读取数据
    sequences, labels = load_data_from_directories(root_dir, data_dirs, args.task)

    # 划分训练集和验证集

    seq_train, seq_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    collate_fn = CollateFunction(args.task)
    # 创建数据集和数据加载器
    if args.mode == 'train':
        train_dataset = WaveformDataset(seq_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        # val_dataset = WaveformDataset(seq_val, y_val)
        # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
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
    else:
        raise ValueError(f"无效的 task 参数: {args.task}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # 加入学习率自动调节
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # Wrap the model for multi-GPU
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model)
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if args.mode == 'train':
        # 训练模型
        train(model, train_loader, seq_val, y_val, criterion, optimizer, device, args.epochs, args.task)
    elif args.mode == 'test':
        if not args.model_path:
            print("测试模式需要指定模型文件路径，请使用 --model_path 参数。")
            exit(1)
        test(model, seq_val, criterion, device, args.model_path, args.task)

"""测试指令
python main.py --mode test --task SW --network CNNRegressor --model_path log/models/SymbolWidth/CNNRegressor/0.1495_60.81_best_model.pth"""

