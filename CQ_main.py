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
from loss.SW_RelativeErrorLoss import RelativeErrorLoss
from loss.CQ_CosineSimilarity import CosineSimilarityLoss
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.CQ_Seq2Seq import Encoder, Decoder

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, train_mode):

    # 设置最佳指标：码元宽度回归任务以最小化损失为目标，调制类型分类任务以最大化准确率为目标，码序列相似度任务以最大化余弦相似度为目标
    best_metric = float('inf') if train_mode == "SW" else 0

    history_train_loss = []
    history_valid_loss = []

    if train_mode == "MT":
        end_path = "ModulationType"
    elif train_mode == "SW":
        end_path = "SymbolWidth"
    elif train_mode == "CQ":
        end_path = "CodeSequence"
    else:
        raise ValueError(f"无效的 train_mode 参数: {train_mode}")
    model_dir = f'log/models/{end_path}/{model.__class__.__name__}'
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        total_train_samples = 0
        for batch_X, seq_lengths, batch_y,label_lengths in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            seq_lengths = seq_lengths.to(device)
            label_lengths = label_lengths.to(device)
            # 调整输入数据的形状
            # batch_X = batch_X.permute(0, 2, 1)  # [batch_size, channels, seq_len]
            print("Input shape:", batch_X.shape)

            optimizer.zero_grad()

            outputs = model(batch_X,seq_lengths,batch_y,label_lengths)
            print("Output shape:", outputs.shape)

            # # 处理输出以匹配标签长度
            # outputs = outputs[:, :max_label_length, :]  # 截取输出到最大标签长度
            # # 创建掩码，确保只计算有效部分
            # mask = (labels != -1).float()
            print(f"labels type: {type(labels)}, shape: {labels.shape if isinstance(labels, torch.Tensor) else 'N/A'}")
            print(f"label_lengths type: {type(label_lengths)}, values: {label_lengths}")

            # 计算余弦相似度损失
            # loss = criterion(outputs, labels.float(), mask)
            loss = criterion(outputs, labels,label_lengths)  # 生成任务

            train_loss += loss.item()
            total_train_samples += batch_X.size(0) # 累加样本数
            loss.backward()
            optimizer.step()
        # 当使用reduce='sum'时，每个批次的损失是所有样本的MSE，因此需要除以总样本数
        # train_loss /= len(train_loader)
        train_loss /= total_train_samples
        history_train_loss.append(train_loss)

        # 验证阶段
        model.eval()
        sum_val_loss = 0.0
        total_samples = 0  # 用于累加样本数量
        if train_mode == "MT":
            correct = 0
            total = 0
        else:
            all_preds = []
            all_targets = []

        with torch.no_grad():
            for batch_X, seq_lengths, batch_y,label_lengths in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                seq_lengths = seq_lengths.to(device)
                label_lengths = label_lengths.to(device)
                # batch_X = batch_X.permute(0, 2, 1)  # [batch_size, channels, seq_len]
                print("Input shape:", batch_X.shape)

                outputs = model(batch_X,seq_lengths)
                # # 处理输出以匹配标签长度
                # max_label_length = label_lengths.max().item()
                # outputs = outputs[:, :max_label_length, :]  # 截取到最大标签长度

                if train_mode == "SW":
                    loss = criterion(outputs, batch_y.view(-1, 1))
                    sum_val_loss += loss.item()  # 累加总损失
                    all_preds.extend(outputs.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
                    total_samples += batch_X.size(0)  # 累加样本数
                elif train_mode == "MT":
                    loss = criterion(outputs, batch_y)
                    sum_val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                elif train_mode == "CQ":
                    # loss = criterion(outputs, labels_embedded,label_lengths)
                    sum_val_loss += loss.item()
        if train_mode == "SW":
            avg_val_loss = sum_val_loss / total_samples  # 计算平均损失
            mse = mean_squared_error(all_targets, all_preds)
            rmse = mean_squared_error(all_targets, all_preds, squared=False)
            mae = mean_absolute_error(all_targets, all_preds)

            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}')
            if mse < best_metric:
                best_metric = mse
                # 保存最佳模型
                torch.save(model.state_dict(), os.path.join(model_dir, f'best_model.pth'))
        elif train_mode == "MT":
            avg_val_loss = sum_val_loss / len(val_loader)  # 对于分类任务，可以保持原有方式
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
            if accuracy > best_metric:
                best_metric = accuracy
                # 保存最佳模型
                torch.save(model.state_dict(), os.path.join(model_dir, f'best_model.pth'))
        elif train_mode == "CQ":
            avg_val_loss = sum_val_loss / len(val_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}')

            # 保存最佳模型
            if avg_val_loss < best_metric:
                best_metric = avg_val_loss
                torch.save(model.state_dict(), os.path.join(model_dir, f'best_model.pth'))
        history_valid_loss.append(avg_val_loss)

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(model_dir,'final_model.pth'))

    # loss图保存路径
    save_path = f'log/save_loss/{end_path}/{model.__class__.__name__}'
    os.makedirs(save_path, exist_ok=True)
    show_plot(history_train_loss, history_valid_loss,
              f"{save_path}/{model.__class__.__name__}_{args.lr}_{args.batch_size}_{best_metric}.png")


def test(model, val_loader, criterion, device, model_path, train_mode):
    # 加载已保存的模型权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型已成功加载: {model_path}")
    else:
        print(f"模型文件不存在: {model_path}")
        return

    model.eval()

    val_loss = 0.0
    total_samples = 0
    if train_mode in ["MT", "CQ"]:
        correct = 0
        total = 0
    else:
        all_preds = []
        all_targets = []

    with torch.no_grad():
        for batch_X, lengths, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            lengths = lengths.to(device)

            batch_X = batch_X.permute(0, 2, 1)  # [batch_size, channels, seq_len]

            outputs = model(batch_X)

            if train_mode == "SW":
                loss = criterion(outputs, batch_y.view(-1, 1))
                val_loss += loss.item()  # 累加总损失
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                total_samples += batch_X.size(0)  # 累加样本数
            else:
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

    if train_mode in ["MT", "CQ"]:
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        print(f'验证集上的损失: {avg_val_loss:.4f}, 准确率: {accuracy:.2f}%')
    elif train_mode == "SW":
        avg_val_loss = val_loss / total_samples
        mse = mean_squared_error(all_targets, all_preds)
        rmse = mean_squared_error(all_targets, all_preds, squared=False)
        mae = mean_absolute_error(all_targets, all_preds)
        print(f'验证集上的损失: {avg_val_loss:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("Train or test the model")
    arg_parser.add_argument("--mode", type=str, default="train", help="train or test")
    arg_parser.add_argument("--train_mode", type=str, default="CQ",
                            help="MT(ModulationType)、SW(SymbolWidth)、CQ(CodeSequence)")
    arg_parser.add_argument("--network", type=str, default="CQ_Seq2Seq",
                            help="选择网络 (例如 CNNClassifier, ResNet)")
    arg_parser.add_argument("--lr", type=float, default=0.0005, help="学习率")
    arg_parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    arg_parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    arg_parser.add_argument("--model_path", type=str, default="",
                            help="模型文件路径，用于测试模式")
    args = arg_parser.parse_args()

    # 指定根目录
    root_dir = 'Dataset'
    data_dirs = ['1', '2', '3', '4']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取数据
    sequences, labels = load_data_from_directories(root_dir, data_dirs, args.train_mode)

    # 划分训练集和验证集
    if args.train_mode == "MT":
        stratify = labels
    else:
        stratify = None

    seq_train, seq_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=stratify
    )
    collate_fn = CollateFunction(args.train_mode)
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
        if args.network == "CQ_Seq2Seq":
            input_dim = 2  # IQ流的输入维度
            output_dim = 16  # 码序列的词汇表大小
            emb_dim = 64  # 嵌入维度
            hidden_dim = 128  # 隐藏层维度
            n_lays = 2  # LSTM层数
            dropout = 0.5  # Dropout概率

            encoder = Encoder(input_dim, hidden_dim, n_lays, dropout).to(device)
            decoder = Decoder(output_dim, emb_dim, hidden_dim, n_lays, dropout).to(device)
            # 创建 Seq2Seq 实例
            model = model_class(encoder, decoder)
        else:
            model = model_class()
    except (ImportError, AttributeError):
        raise ValueError(f"指定的网络 {args.network} 无效或不存在")

    # 定义损失函数和优化器
    if args.train_mode == "MT":
        criterion = nn.CrossEntropyLoss()
    elif args.train_mode == "SW":
        criterion = RelativeErrorLoss()
    elif args.train_mode == "CQ":
        criterion = CosineSimilarityLoss(model.decoder)
    else:
        raise ValueError(f"无效的 train_mode 参数: {args.train_mode}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.to(device)

    if args.mode == 'train':
        # 训练模型
        train(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, args.train_mode)
    elif args.mode == 'test':
        if not args.model_path:
            print("测试模式需要指定模型文件路径，请使用 --model_path 参数。")
            exit(1)
        test(model, val_loader, criterion, device, args.model_path, args.train_mode)
