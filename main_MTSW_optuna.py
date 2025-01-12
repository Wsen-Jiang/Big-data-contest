import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse
import importlib
from utils.dataset import load_data_from_directories, WaveformDataset, CollateFunction
from utils.utils import show_plot, set_random_seed
from Criterion.SW_RelativeErrorLoss import RelativeErrorLoss
import numpy as np
from tqdm import tqdm
from utils.utils import show_plot
import matplotlib.pyplot as plt
import optuna
import math
import copy

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

def train(model, train_loader, seq_val, y_val, criterion, optimizer, device, num_epochs, task):
    # 设置最佳指标：码元宽度回归任务以最小化损失为目标，调制类型分类任务以最大化准确率为目标，码序列相似度任务以最大化余弦相似度为目标
    best_metric = 0
    best_epoch = 0
    best_model_weights = copy.deepcopy(model.state_dict())  # 用于保存最优模型的权重

    history_train_loss = []
    history_valid_loss = []

    if task == "MT":
        end_path = "ModulationType"
    elif task == "SW":
        end_path = "SymbolWidth"
    else:
        raise ValueError(f"无效的 task 参数: {task}")

    model_name = model.module.__class__.__name__ if isinstance(model, nn.DataParallel) else model.__class__.__name__
    model_dir = f'log/models/{end_path}/{model_name}'
    os.makedirs(model_dir, exist_ok=True)
    # record val testing log after each epoch training
    f = open(os.path.join(model_dir, 'training.log'), 'w')

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_X, seq_lengths, batch_y in tqdm(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            seq_lengths = seq_lengths.to(device)

            # 调整输入数据的形状
            # batch_X = batch_X.permute(0, 2, 1)  # [batch_size, channels, seq_len]

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.zero_grad()

            outputs = model(batch_X)

            if task == "SW":
                loss = criterion(outputs, batch_y.view(-1, 1))  # 回归任务
            else:
                loss = criterion(outputs, batch_y)  # 分类任务

            train_loss += loss.item()

            if task == "MT":
                # 计算分类准确率
                _, predicted = torch.max(outputs.data, 1)
                total_train += batch_y.size(0)
                correct_train += (predicted == batch_y).sum().item()

            loss.backward()
            optimizer.step()

        # 计算训练集损失和准确率
        train_loss /= len(train_loader)
        history_train_loss.append(train_loss)

        # 验证阶段
        model.eval()
        sum_val_loss = 0.0
        if task == "MT":
            correct = 0
        else:
            mean_score = 0
        with torch.no_grad():
            for seq, val in zip(seq_val, y_val):
                seq = seq.to(device)  # 将seq移动到device
                val = val.clone().detach().to(device)

                seq = seq.unsqueeze(0)

                # 单个验证样本的模型输出
                seq = seq.permute(0, 2, 1)
                outputs = model(seq).item()

                if task == "SW":
                    if not math.isnan(outputs):
                        predict_SW = round(outputs, 2)
                        predict_SW = round(predict_SW / 0.05) * 0.05
                    else:
                        print("Warning: predict_SW is NaN after outputs.item(). Assigning default.")
                        predict_SW = 0.0  # 默认值

                    val = round(val.item(), 2)
                    score_error = np.abs(predict_SW - val)
                    score = calculate_score(score_error)
                    mean_score += score

                else:
                    val = val.unsqueeze(0)
                    valid_loss = criterion(outputs, val)
                    sum_val_loss += valid_loss
                    modulation_type = torch.argmax(outputs, dim=1) + 1  # 调制类型预测
                    correct += (modulation_type == (val+1))


        if task == "SW":
            mean_score /= len(seq_val)
            print(f'Epoch [{epoch + 1}/{num_epochs}], , Train Loss: {train_loss:.4f},  Validation Mean Score: {mean_score:.2f}')
            f.write(f'Epoch [{epoch + 1}/{num_epochs}], , Train Loss: {train_loss:.4f}, Validation Mean Score: {mean_score:.2f}' + '\n')
            if mean_score > best_metric:
                best_metric = mean_score
                best_epoch = epoch
                # 保存最佳模型参数
                best_model_weights = copy.deepcopy(model.state_dict())
                print(f"在第{epoch + 1}轮，验证集上最优得分:{best_metric}")
        else:
            # 验证集准确率
            accuracy = 100 * correct / len(seq_val)
            accuracy = accuracy.item()
            # 训练集准确率
            accuracy_train = 100 * correct_train / total_train
            # 验证集损失
            valid_mean_loss = sum_val_loss / len(y_val)
            history_valid_loss.append(valid_mean_loss)

            print(f'Epoch [{epoch + 1}/{num_epochs}],  Train Loss: {train_loss:.4f}, Train Accuracy: {accuracy_train:.2f}%, Validation Loss: {valid_mean_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')
            f.write(f'Epoch [{epoch + 1}/{num_epochs}],  Train Loss: {train_loss:.4f}, Train Accuracy: {accuracy_train:.2f}%, Validation Loss: {valid_mean_loss:.4f}, Validation Accuracy: {accuracy:.2f}%' + '\n')
            if accuracy > best_metric:
                best_metric = accuracy
                best_epoch = epoch
                # 保存最佳模型
                best_model_weights = copy.deepcopy(model.state_dict())
                print(f"在第{epoch + 1}轮，验证集上最优得分:{best_metric}")
    print(f"在第{best_epoch + 1}轮，验证集上最优得分:{best_metric}")

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pth'))
    f.close()

    # loss图保存路径
    save_path = f'log/save_loss/{end_path}/{model.__class__.__name__}'
    os.makedirs(save_path, exist_ok=True)
    show_plot(history_train_loss, history_valid_loss,
              f"{save_path}/{model.__class__.__name__}_{args.lr}_{args.batch_size}_{best_metric}.png")
    return best_metric, model_dir, best_model_weights  #返回最优指标

def test(model, seq_val, y_val, device, model_path, task):
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        return False

    try:
        # 加载权重
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)

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
        print(f"[成功] 模型已成功加载: {model_path}")

    except RuntimeError as e:
        print(f"[错误] 模型加载失败: {e}")
        return False

    model.eval()

    if task == "MT":
        correct = 0
        list = []
    else:
        Avg_Score = 0

    with torch.no_grad():
        for seq, val in zip(seq_val, y_val):
            val = val.clone().detach().to(device)
            seq = seq.clone().detach() if isinstance(seq, torch.Tensor) else torch.tensor(seq)
            seq = seq.unsqueeze(0).permute(0, 2, 1).to(device)  # 数据移动到设备
            output = model(seq)

            if task == "MT":
                logits = torch.nn.functional.softmax(output, dim=1)
                # 选出最高的概率
                max_score = torch.max(logits).item()
                modulation_type = torch.argmax(output, dim=1) + 1
                if (modulation_type == (val+1)).item():
                    correct += 1
                    list.append(max_score)
            else:
                # 预测码元宽度
                predict_SW = round(output.item(), 2)
                predict_SW = round(predict_SW/0.05)*0.05
                label = round(val.item(), 2)
                score_error = np.abs(predict_SW - label)
                score = calculate_score(score_error)
                Avg_Score += score

    if task == "MT":
        accuracy = 100 * correct / len(seq_val)
        print(f'准确率: {accuracy:.3f}%')

        # 绘制list直方图
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

    elif task == "SW":
        Avg_Score /= len(seq_val)
        print(f"总得分: {Avg_Score:.2f}")

def objective(trial, model_class, seq_train, seq_val, y_train, y_val, criterion, device, collate_fn, task):
    """
    objective 函数：
      1. 定义超参数搜索空间
      2. 读取数据并创建 DataLoader
      3. 创建模型、定义损失函数和优化器
      4. 训练并验证，返回验证集的指标（这里以 SymbolWidth 的平均得分 mean_score 为例）
    """
    # 超参数搜索空间
    lr = trial.suggest_float("lr", 5e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
    epochs = trial.suggest_int("epochs", 150, 200, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    print(f"lr: {lr}, batch_size: {batch_size}, epochs: {epochs}, weight_decay: {weight_decay}")

    # 创建数据集和数据加载器
    train_dataset = WaveformDataset(seq_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 初始化模型和优化器
    model = model_class().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 训练模型
    best_score, model_dir, best_model_weights = train(
        model,
        train_loader,
        seq_val,
        y_val,
        criterion,
        optimizer,
        device,
        epochs,
        task
    )
    save_name = f'{trial.number:.0f}_{task}_{best_score}_best_model.pth'
    torch.save(best_model_weights, os.path.join(model_dir, save_name))
    print(f"[INFO] 本 trial={trial.number} 的最佳模型已保存到: {save_name}")
    # 返回验证集的指标
    return best_score

if __name__ == "__main__":

    set_random_seed(42)
    arg_parser = argparse.ArgumentParser("Train or test the model")
    arg_parser.add_argument("--mode", type=str, default="train", help="train or test")
    arg_parser.add_argument("--task", type=str, default="SW",
                            help="MT(ModulationType)、SW(SymbolWidth)")
    arg_parser.add_argument("--network", type=str, default="ResBlock_Regressor",
                            help="选择网络 (例如 CNNClassifier, ResNet, CNN_LSTM_Classifier)")
    arg_parser.add_argument("--lr", type=float, default=0.003, help="学习率")
    arg_parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    arg_parser.add_argument("--batch_size", type=int, default=2048, help="批次大小")
    arg_parser.add_argument("--model_path", type=str,
                            default="/mnt/data/JWS/Big-data-contest/log/models/SymbolWidth/best_model_64.72.pth",
                            help="模型文件路径，用于测试模式")
    arg_parser.add_argument("--auto_tune", action="store_true",
                            default=True, help="是否进行自动调参模式；如指定，则使用Optuna进行调参")
    args = arg_parser.parse_args()

    # 指定根目录
    root_dir = './train_data'
    data_dirs = ['8APSK', '8PSK', '8QAM', '16APSK','16QAM','32APSK','32QAM','BPSK','MSK','QPSK']
    # data_dirs = ['8APSK']


    sequences, labels = load_data_from_directories(root_dir, data_dirs, args.task)

    # 划分训练集和验证集

    seq_train, seq_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 创建数据集和数据加载器
    collate_fn = CollateFunction(args.task)

    # 动态导入或选择网络模型
    try:
        module = importlib.import_module(f"models.{args.network}")
        model_class = getattr(module, args.network)
    except (ImportError, AttributeError):
        raise ValueError(f"指定的网络 {args.network} 无效或不存在")

    # 使用 DataParallel
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model, device_ids=[1, 2, 3])  # 定义损失函数和优化器

    # 定义损失函数
    if args.task == "MT":
        criterion = nn.CrossEntropyLoss()
    elif args.task == "SW":
        criterion = RelativeErrorLoss()
    else:
        raise ValueError(f"无效的 task 参数: {args.task}")

    if args.auto_tune:
        # 使用 Optuna 进行超参数调优
        study = optuna.create_study(direction="maximize", study_name=args.task+"_"+args.network+"_Optimization")
        study.optimize(
            lambda trial: objective(
                trial=trial,
                model_class = model_class,
                seq_train = seq_train,
                seq_val=seq_val,
                y_train = y_train,
                y_val = y_val,
                criterion=criterion,
                device=device,
                collate_fn=collate_fn,
                task = args.task
            ),
            n_trials=20  # 迭代次数
        )
        print("调参结束，最佳参数：")
        print(study.best_params)
        print("最佳得分：")
        print(study.best_value)
        exit(0)
    # 根据参数训练/测试模型
    else:
        # 创建数据集和数据加载器
        if args.mode == 'train':
            train_dataset = WaveformDataset(seq_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
            # 使用原始IQ长度作为验证集
            # val_dataset = WaveformDataset(seq_val, y_val)
            # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        elif args.mode == 'test':
            # val_dataset = WaveformDataset(seq_val, y_val)
            # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
            pass
        else:
            raise ValueError("无效的模式，请选择 'train' 或 'test'")
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

        if args.mode == 'train':
            # 训练模型
            train(model, train_loader, seq_val, y_val, criterion, optimizer, device, args.epochs, args.task)
        elif args.mode == 'test':
            if not args.model_path:
                print("测试模式需要指定模型文件路径，请使用 --model_path 参数。")
                exit(1)
            test(model, seq_val, y_val, device, args.model_path, args.task)


