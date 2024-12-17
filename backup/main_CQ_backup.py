import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse
import importlib

from dataset import load_data_from_directories, WaveformDataset, CollateFunction
from utils.utils import show_plot
from Criterion.CosineSimilarityLoss import CosineSimilarityLoss
import numpy as np
from utils.cosinesimilarity import cosine_similarity
from models.CQ_Seq2Seq import Encoder, Decoder

# 计算码元宽度得分
def calculate_score(cosine_similarity):
    if cosine_similarity >= 0.95:
        return 100
    elif cosine_similarity <= 0.7:
        return 0
    return (cosine_similarity-0.7)*400

def output_process(outputs,labels):
    ## 去除BOS
    #labels = labels[:,1:]
    output_seq_len = outputs.size(1)
    target_seq_len = labels.size(1)
    max_len = max(output_seq_len, target_seq_len)

    # 如果 outputs 的长度不足 max_len，则在末尾填充
    if output_seq_len < max_len:
        padding_size = max_len - output_seq_len
        padding = torch.full((outputs.size(0), padding_size, outputs.size(2)), fill_value=0,
                             device=outputs.device)
        outputs = torch.cat([outputs, padding], dim=1)

    # 如果 labels 的长度不足 max_len，则在末尾填充
    if target_seq_len < max_len:
        padding_size = max_len - target_seq_len
        padding = torch.full((labels.size(0), padding_size), fill_value=18,
                             device=labels.device)
        labels = torch.cat([labels, padding], dim=1)

    return outputs, labels

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs,task):

    # 设置最佳指标：码元宽度回归任务以最小化损失为目标，调制类型分类任务以最大化准确率为目标，码序列相似度任务以最大化余弦相似度为目标
    best_metric = 0

    history_train_loss = []
    history_valid_loss = []


    end_path = "CodeSequence"

    model_dir = f'log/models/{end_path}/{model.__class__.__name__}'
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0

        for batch_X, seq_lengths, batch_y,label_lengths in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            seq_lengths,label_lengths = seq_lengths.to(device),label_lengths.to(device)

            # 调整输入数据的形状
           #  batch_X = batch_X.permute(0, 2, 1)  # [batch_size, channels, seq_len]

            optimizer.zero_grad()

            outputs = model(batch_X, seq_lengths, batch_y, label_lengths)
            # for i in range(len(outputs)):
            #     print(outputs[i])
            outputs, labels = output_process(outputs, batch_y)
            # for i in range(outputs.size(0)):
            #     print(f'output:{i}: {outputs[i].argmax(-1)}')
            #     print(f'labels{i}: {labels[i]}')
            # 展平张量
            outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * max_len, output_dim)
            labels = labels.view(-1)  # (batch_size * max_len)
            # 定义损失函数，忽略填充符的索引
            loss = criterion(outputs, labels)

            # loss = criterion(outputs, labels, label_lengths)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')
        history_train_loss.append(train_loss)

        # 验证阶段
        model.eval()
        sum_val_loss = 0.0

        with torch.no_grad():
            for batch_X, seq_lengths, batch_y,label_lengths in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                seq_lengths,label_lengths = seq_lengths.to(device), label_lengths.to(device)

                # batch_X = batch_X.permute(0, 2, 1)  # [batch_size, channe ls, seq_len]

                outputs = model(batch_X,seq_lengths,batch_y,label_lengths, use_teacher_forcing=False)
                # 填充操作
                outputs, labels = output_process(outputs, batch_y)

                # 展平张量
                outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * max_len, output_dim)
                labels = labels.view(-1)  # (batch_size * max_len)

                # 计算余弦相似度
                consimilarity = cosine_similarity(outputs.argmax(dim=-1).cpu().numpy(), labels.cpu().numpy())

                # 取整
                # outputs = torch.round(outputs).to(torch.int64)
                loss = criterion(outputs, labels)
                sum_val_loss += loss.item()

        avg_val_loss = sum_val_loss / len(val_loader)  # 计算平均损失

        print(f'Epoch [{epoch + 1}/{num_epochs}],    Val Loss: {avg_val_loss:.4f},    cosinesimilarity: {consimilarity}')
        if avg_val_loss > best_metric:
            best_metric = avg_val_loss
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

    sum_val_loss = 0.0

    with torch.no_grad():
        for batch_X, seq_lengths, batch_y, label_lengths in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            seq_lengths,label_lengths = seq_lengths.to(device),label_lengths.to(device)

            # batch_X = batch_X.permute(0, 2, 1)  # [batch_size, channels, seq_len]

            outputs = model(batch_X,seq_lengths,batch_y,label_lengths,use_teacher_forcing=False)

            outputs = torch.round(outputs).to(torch.int64)
            loss = criterion(outputs, batch_y, label_lengths)
            sum_val_loss += loss.item()

    avg_val_loss = sum_val_loss / len(val_loader)  # 计算平均损失
    cosine_similarity = 1 - avg_val_loss
    print(f'Val Loss: {avg_val_loss:.4f}, CosineSimilarity:{cosine_similarity:.4f}')
    print(f'CodeSequence Score: {calculate_score(cosine_similarity):.2f}')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("Train or test the model")
    arg_parser.add_argument("--mode", type=str, default="train", help="train or test")
    arg_parser.add_argument("--task", type=str, default="CQ",
                            help="only CQ")
    arg_parser.add_argument("--network", type=str, default="CQ_Seq2Seq",
                            help="选择网络")
    arg_parser.add_argument("--lr", type=float, default=0.0005, help="学习率")
    arg_parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    arg_parser.add_argument("--batch_size", type=int, default=1024, help="批次大小")
    arg_parser.add_argument("--model_path", type=str, default="",
                            help="模型文件路径，用于测试模式")
    arg_parser.add_argument("--padding_mode", type=str, default="zero",
                            help="optional: zero, cons")
    args = arg_parser.parse_args()

    # 指定根目录
    root_dir = 'train_data'
    data_dirs = ['8APSK', '8PSK', '8QAM', '16APSK','16QAM','32APSK','32QAM','BPSK','MSK','QPSK']
    # data_dirs = ['8APSK']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 读取数据
    sequences, labels = load_data_from_directories(root_dir, data_dirs, args.task)

    # 划分训练集和验证集
    seq_train, seq_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    collate_fn = CollateFunction(args.task, args.padding_mode)
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
            output_dim = 35  # 码序列的词汇表大小
            emb_dim = 64  # 嵌入维度
            hidden_dim = 128  # 隐藏层维度
            n_lays = 2  # LSTM层数
            dropout = 0.5  # Dropout概率
            vocab = {
                "<BOS>": 16,  # Begin of Sequence 起始符
                "<EOS>": 17,  # End of Sequence 结束符
                "<PAD>": 18,
                "0": 0,
                "1": 1,
                "2": 2,
                "3": 3,
                "4": 4,
                "5": 5,
                "6": 6,
                "7": 7,
                "8": 8,
                "9": 9,
                "10": 10,
                "11": 11,
                "12": 12,
                "13": 13,
                "14": 14,
                "15": 15
            }
            encoder = Encoder(input_dim, hidden_dim, n_lays, dropout).to(device)
            decoder = Decoder(output_dim, emb_dim, hidden_dim, n_lays, dropout).to(device)
            # 创建 Seq2Seq 实例
            model = model_class(encoder, decoder, vocab)
        else:
            model = model_class()
    except (ImportError, AttributeError):
        raise ValueError(f"指定的网络 {args.network} 无效或不存在")

    # 定义损失函数和优化器
    #criterion = CosineSimilarityLoss()
    criterion = nn.CrossEntropyLoss(ignore_index=18)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
python main_backup.py --mode test --task SW --network CNNRegressor --model_path log/models/SymbolWidth/CNNRegressor/0.1495_60.81_best_model.pth"""
