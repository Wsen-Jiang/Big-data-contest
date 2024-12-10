import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
from sklearn.model_selection import train_test_split

from utils.dataset import load_data_from_directories, WaveformDataset, CollateFunction
from utils.utils import show_plot
from utils.cosinesimilarity import cosine_similarity_seq
from models.IQToCodeSeqModel import IQToCodeSeqModel  # 导入 IQToCodeSeqModel
import logging

# 配置日志格式和级别
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),  # 将日志写入文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

logger = logging.getLogger(__name__)


# 计算码元宽度得分
def calculate_score(cosine_similarity):
    if cosine_similarity >= 0.95:
        return 100
    elif cosine_similarity <= 0.7:
        return 0
    return (cosine_similarity - 0.7) * 400


def decode_sequence(indices, vocab):
    # 定义反向词汇表
    idx_to_token = {idx: token for token, idx in vocab.items()}
    tokens = []
    for idx in indices:
        token = idx_to_token.get(idx, "<UNK>")
        if token in ["<PAD>", "<BOS>", "<EOS>"]:
            continue  # 忽略特殊标记
        tokens.append(token)
    return tokens


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, vocab_size):
    # 设置最佳指标
    best_metric = -float('inf')  # 余弦相似度越大越好

    history_train_loss = []
    history_valid_similarity = []

    end_path = "CodeSequence"

    model_dir = f'log/models/{end_path}/{model.__class__.__name__}'
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        batch_count = 0
        teacher_forcing_ratio = max(0.5, 1.0 - epoch / num_epochs)
        # logger.info(
        #     f"Epoch [{epoch + 1}/{num_epochs}] - Starting training with teacher_forcing_ratio={teacher_forcing_ratio:.2f}")

        for batch_idx, (batch_X, seq_lengths, batch_y, label_lengths) in enumerate(train_loader, 1):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            seq_lengths, label_lengths = seq_lengths.to(device), label_lengths.to(device)

            # 调整输入数据的形状
            batch_X = batch_X.permute(0, 2, 1)  # [batch_size, channels, seq_len]

            optimizer.zero_grad()
            outputs = model(batch_X, seq_lengths, batch_y, teacher_forcing_ratio)
            # outputs: (batch_size, tgt_seq_length, vocab_size)

            # 计算损失
            loss = criterion(outputs.reshape(-1, vocab_size), batch_y.view(-1))

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            batch_count += 1
            # 每50个批次记录一次训练损失
            if batch_count % 50 == 0:
                avg_loss = train_loss / batch_count
                logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_count}], Train Loss: {avg_loss:.4f}')


        Average_train_loss = train_loss / batch_count
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}] - Average Train Loss: {Average_train_loss:.4f}')
        history_train_loss.append(Average_train_loss)

        # 验证阶段
        model.eval()
        total_cosine_similarity = 0.0
        val_batch_count = 0
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}] - Starting validation")

        with torch.no_grad():
            for batch_idx, (batch_X, seq_lengths, batch_y, label_lengths) in enumerate(val_loader, 1):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                seq_lengths, label_lengths = seq_lengths.to(device), label_lengths.to(device)

                # 调整输入数据的形状
                batch_X = batch_X.permute(0, 2, 1)  # [batch_size, channels, seq_len]

                # 生成序列，不使用教师强制
                generated_sequences = model(batch_X, seq_lengths)
                # generated_sequences: (batch_size, max_seq_length)

                # 计算余弦相似度
                # 由于第一个序列是 <BOS>，因此不计算
                consimilarity = cosine_similarity_seq(generated_sequences[:, 1:], batch_y, label_lengths)
                total_cosine_similarity += consimilarity

                val_batch_count += 1

                # 每10个批次计算一次余弦相似度
                if val_batch_count % 10 == 0:
                    avg_cosine = total_cosine_similarity / val_batch_count
                    logger.info(
                        f'Epoch [{epoch + 1}/{num_epochs}], Val Batch [{val_batch_count}], Cosine Similarity: {avg_cosine:.4f}')


        # 计算每个epoch的平均余弦相似度

        avg_cosine_similarity = total_cosine_similarity / val_batch_count
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}] - Average Val CosineSimilarity: {avg_cosine_similarity:.4f}')
        history_valid_similarity.append(avg_cosine_similarity)


        # 保存最佳模型基于余弦相似度
        if avg_cosine_similarity > best_metric:
            best_metric = avg_cosine_similarity
            torch.save(model.state_dict(), os.path.join(model_dir, f'best_model.pth'))
            logger.info(f'Best model saved with Cosine Similarity: {best_metric:.4f}')

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pth'))
    logger.info('Final model saved.')

    # loss图保存路径
    save_path = f'log/save_loss/{end_path}/{model.__class__.__name__}'
    os.makedirs(save_path, exist_ok=True)
    show_plot(history_train_loss, history_valid_similarity,
              f"{save_path}/{model.__class__.__name__}.png")
    logger.info('Training and validation plots saved.')


def test(model, val_loader, criterion, device, model_path, vocab):
    # 加载已保存的模型权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"模型已成功加载: {model_path}")
    else:
        logger.error(f"模型文件不存在: {model_path}")
        return

    model.eval()

    sum_val_loss = 0.0
    total_cosine_similarity = 0.0
    val_batch_count = 0

    generated_sequences_all = []
    labels_all = []

    logger.info("开始测试...")

    with torch.no_grad():
        for batch_idx, (batch_X, seq_lengths, batch_y, label_lengths) in enumerate(val_loader, 1):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            seq_lengths, label_lengths = seq_lengths.to(device), label_lengths.to(device)

            # 调整输入数据的形状
            batch_X = batch_X.permute(0, 2, 1)  # [batch_size, channels, seq_len]

            # 生成序列，不使用教师强制
            generated_sequences = model(batch_X, seq_lengths)
            # generated_sequences: (batch_size, max_seq_length)

            # 收集生成序列和标签
            generated_sequences_all.append(generated_sequences.cpu())
            labels_all.append(batch_y.cpu())

            # 计算余弦相似度
            consimilarity = cosine_similarity_seq(generated_sequences[:,1:], batch_y, label_lengths)
            total_cosine_similarity += consimilarity

            val_batch_count += 1

            # 每20个批次记录一次余弦相似度
            if val_batch_count % 20 == 0:
                avg_cosine = total_cosine_similarity / val_batch_count
                logger.info(f'Test Batch [{val_batch_count}], Cosine Similarity: {avg_cosine:.4f}')

    # 拼接所有批次
    generated_sequences_all = torch.cat(generated_sequences_all, dim=0)  # (total_samples, max_seq_length)
    labels_all = torch.cat(labels_all, dim=0)  # (total_samples, max_seq_length)

    # 解码生成的序列
    decoded_generated = [decode_sequence(seq, vocab) for seq in generated_sequences_all]
    decoded_labels = [decode_sequence(seq, vocab) for seq in labels_all]

    # 计算最终的余弦相似度
    avg_cosine_similarity = total_cosine_similarity / val_batch_count
    final_score = calculate_score(avg_cosine_similarity)
    logger.info(f'Final Val CosineSimilarity: {avg_cosine_similarity:.4f}')
    logger.info(f'CodeSequence Score: {final_score:.2f}')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("Train or test the model")
    arg_parser.add_argument("--mode", type=str, default="train", help="train or test")
    arg_parser.add_argument("--task", type=str, default="CQ", help="only CQ")
    arg_parser.add_argument("--network", type=str, default="IQToCodeSeqModel", help="选择网络: IQToCodeSeqModel")
    arg_parser.add_argument("--lr", type=float, default=0.0001, help="学习率")
    arg_parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    arg_parser.add_argument("--batch_size", type=int, default=1024, help="批次大小")
    arg_parser.add_argument("--model_path", type=str, default="", help="模型文件路径，用于测试模式")
    args = arg_parser.parse_args()

    # 定义词汇表
    vocab = {
        "<PAD>": 0, "<BOS>": 1, "<EOS>": 2,
        "0": 3, "1": 4, "2": 5, "3": 6, "4": 7, "5": 8, "6": 9, "7": 10, "8": 11, "9": 12, "10": 13,
        "11": 14, "12": 15, "13": 16, "14": 17, "15": 18, "16": 19, "17": 20, "18": 21, "19": 22, "20": 23,
        "21": 24, "22": 25, "23": 26, "24": 27, "25": 28, "26": 29, "27": 30, "28": 31, "29": 32, "30": 33,
        "31": 34
    }
    vocab_size = len(vocab)

    # 指定根目录
    root_dir = "/mnt/data/JWS/Big-data-contest/train_data/"
    # root_dir = "/home/JWS/Big_data_contest/train_data/"
    data_dirs = ['8APSK', '8PSK', '8QAM', '16APSK', '16QAM', '32APSK', '32QAM', 'BPSK', 'MSK', 'QPSK']
    # data_dirs = ['8APSK']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取数据
    sequences, labels = load_data_from_directories(root_dir, data_dirs, args.task)

    # 将标签映射到词汇表索引
    mapped_labels = []
    for seq in labels:
        mapped_seq = [vocab.get(str(token.item()), vocab["<PAD>"]) for token in seq]
        mapped_labels.append(mapped_seq)

    # 划分训练集和验证集
    seq_train, seq_val, y_train, y_val = train_test_split(
        sequences, mapped_labels, test_size=0.2, random_state=42
    )
    collate_fn = CollateFunction(args.task, vocab)

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

    # 实例化 IQToCodeSeqModel
    if args.network == "IQToCodeSeqModel":
        model = IQToCodeSeqModel(
            vocab_size=vocab_size,
            bos_idx=vocab["<BOS>"],  # 传递 <BOS> 的索引
            embed_dim=128,
            num_heads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            max_seq_length=450
        ).to(device)

        # 使用 DataParallel
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1, 2])
            print(f"使用 {torch.cuda.device_count()} 张 GPU")

    else:
        raise ValueError(f"指定的网络 {args.network} 无效或不存在")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == 'train':
        # 训练模型
        train(model, train_loader, val_loader, criterion, optimizer, device, args.epochs,vocab_size)

    elif args.mode == 'test':
        if not args.model_path:
            print("测试模式需要指定模型文件路径，请使用 --model_path 参数。")
            exit(1)
        test(model, val_loader, criterion, device, args.model_path,vocab)
