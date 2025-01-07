import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from models.CQ_LSTM import CQ_LSTM

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
from sklearn.model_selection import train_test_split
from Criterion.CosineSimilarityLoss import CosineSimilarityLoss
from utils.dataset import load_data_from_directories, WaveformDataset, CollateFunction
from utils.dataset_old import CollateFunction as CollateFunction_old
from utils.utils import show_plot
from utils.cosinesimilarity import cosine_similarity_seq
from models.IQToCodeSeqModel import IQToCodeSeqModel
from models.CQ_CNNLSTMAttention import CQ_CNNLSTMAttention
from models.ResBlock_CodeSequence import ResBlock_CodeSequence
import logging
import random
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
# 计算码元宽度得分

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


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, vocab_size):
    # 设置最佳指标
    best_metric = -float('inf')  # 余弦相似度越大越好

    history_train_loss = []
    history_valid_similarity = []

    end_path = "CodeSequence"

    model_dir = f'log/models/{end_path}/{model.module.__class__.__name__}' if isinstance(model, nn.DataParallel) else f'log/models/{end_path}/{model.__class__.__name__}'

    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        batch_count = 0
        teacher_forcing_ratio = max(0.5, 1.0 - epoch / num_epochs)
        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}] - Starting training with teacher_forcing_ratio={teacher_forcing_ratio:.2f}")

        for batch_X, seq_lengths, batch_y, label_lengths in tqdm(train_loader):
            # batch_X (batch_size, channels, seq_len)
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            seq_lengths = seq_lengths.to(device)

            # 调整输入数据的形状
            # batch_X = batch_X.permute(0, 2, 1)

            optimizer.zero_grad()
            # outputs = model(batch_X, seq_lengths, batch_y, teacher_forcing_ratio)
            outputs = model(batch_X, seq_lengths, batch_y)
            # outputs: (batch_size, tgt_seq_length, vocab_size)

            # 计算损失,展平并忽略掉 <PAD> 的损失
            loss = criterion(outputs.reshape(-1, vocab_size), batch_y.view(-1))
            # loss = criterion(outputs, batch_y)
            train_loss += loss.item()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step(loss)

            batch_count += 1
            # 每50个批次记录一次训练损失
            if batch_count % 50 == 0:
                avg_loss = train_loss / batch_count
                logger.info(f'Epoch [{epoch + 1}/{num_epochs}],  前{batch_count}Batch的损失, Train Loss: {avg_loss:.4f}')


        Average_train_loss = train_loss / batch_count
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}] - Average Train Loss: {Average_train_loss:.4f}')
        history_train_loss.append(Average_train_loss)

        # 验证阶段
        model.eval()
        total_cosine_similarity = 0.0
        val_batch_count = 0
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}] - Starting validation")

        with torch.no_grad():
            for batch_X, seq_lengths, batch_y, label_lengths in tqdm(val_loader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                seq_lengths, label_lengths = seq_lengths.to(device), label_lengths.to(device)

                # 调整输入数据的形状
                batch_X = batch_X.permute(0, 2, 1)  # [batch_size, channels, seq_len]

                # 生成序列，不使用教师强制
                # generated_sequences = model(batch_X, seq_lengths)
                generated_sequences = model(batch_X, seq_lengths)
                # generated_sequences: (batch_size, max_seq_length)

                # 由于第一个序列是 <BOS>，因此不计算
                generated_sequences = generated_sequences[:, 1:]
                # 检测是否全部为 0，并替换为全 1
                if torch.all(generated_sequences[:5] == 0):  # 检查前5个元素是否全为0
                    generated_sequences = torch.ones_like(generated_sequences)  # 将整个张量设为1

                # 计算余弦相似度
                consimilarity = cosine_similarity_seq(generated_sequences, batch_y, label_lengths)
                total_cosine_similarity += consimilarity

                val_batch_count += 1

                # 每10个批次计算一次余弦相似度
                if val_batch_count % 10 == 0:
                    avg_cosine = total_cosine_similarity / val_batch_count
                    logger.info(
                        f'Epoch [{epoch + 1}/{num_epochs}], 前{val_batch_count}的Val Batch Cosine Similarity: {avg_cosine:.4f}')

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


def test(model, seq_val, y_val, device, model_path, vocab):
    # 加载已保存的模型权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"模型已成功加载: {model_path}")
    else:
        logger.error(f"模型文件不存在: {model_path}")
        return

    model.eval()

    total_cosine_similarity = 0.0
    val_batch_count = 0

    logger.info("开始测试...")

    with torch.no_grad():
        for seq, val in zip(seq_val, y_val):
            seq = seq.to(device)  # 将seq移动到device
            val = val.clone().detach().to(device)
            seq = seq.unsqueeze(0)  # 增加一个 batch_size 维度
            seq_length, label_length = len(seq), len(val)

            # 生成序列，不使用教师强制
            generated_sequences = model(seq, seq_length)
            # generated_sequences: (batch_size, max_seq_length)

            # 计算余弦相似度
            consimilarity = cosine_similarity_seq(generated_sequences[:,1:], val, label_length)
            total_cosine_similarity += consimilarity

            val_batch_count += 1

            # 每20个批次记录一次余弦相似度
            if val_batch_count % 20 == 0:
                avg_cosine = total_cosine_similarity / val_batch_count
                logger.info(f'Test Batch [{val_batch_count}], Cosine Similarity: {avg_cosine:.4f}')


    # 计算最终的余弦相似度
    avg_cosine_similarity = total_cosine_similarity / val_batch_count
    final_score = calculate_score(avg_cosine_similarity)
    logger.info(f'Final Val CosineSimilarity: {avg_cosine_similarity:.4f}')
    logger.info(f'CodeSequence Score: {final_score:.2f}')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("Train or test the model")
    arg_parser.add_argument("--mode", type=str, default="train", help="train or test")
    arg_parser.add_argument("--task", type=str, default="CQ", help="only CQ")
    arg_parser.add_argument("--network", type=str, default="CQ_CNNLSTMAttention", help="选择网络: CQ_CNNLSTMAttention")
    arg_parser.add_argument("--lr", type=float, default=0.0005, help="学习率")
    arg_parser.add_argument("--epochs", type=int, default=150, help="训练轮数")
    arg_parser.add_argument("--batch_size", type=int, default=2048, help="批次大小")
    arg_parser.add_argument("--model_path", type=str, default="", help="模型文件路径，用于测试模式")
    args = arg_parser.parse_args()
    set_random_seed(42)

    # 定义词汇表
    # vocab = {
    #     0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14,
    #     15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27,
    #     28: 28, 29: 29, 30: 30, 31: 31, "<PAD>": 32, "<BOS>": 33, "<EOS>": 34
    # }
    vocab = {
        0: 0, 1: 1, "<PAD>": 2, "<BOS>": 3 }
    vocab_size = len(vocab)

    # 指定根目录
    # root_dir = "/home/JWS/Big-data-contest/train_data/"
    root_dir = "/mnt/data/JWS/Big-data-contest/train_data/"
    # data_dirs = ['8APSK', '8PSK', '8QAM', '16APSK', '16QAM', '32APSK', '32QAM', 'BPSK', 'MSK', 'QPSK']
    data_dirs = ['BPSK', 'MSK']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 读取数据
    sequences, labels = load_data_from_directories(root_dir, data_dirs, args.task)

    # 划分训练集和验证集
    seq_train, seq_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    # all_labels = torch.cat(y_train)

    # # 统计每个类别的出现次数
    # class_counts = torch.bincount(all_labels, minlength=vocab_size)
    #
    # print("Class Counts:", class_counts)

    collate_fn = CollateFunction(args.task, vocab)
    collate_fn_old = CollateFunction_old(args.task, vocab)

    # 创建数据集和数据加载器
    if args.mode == 'train':
        train_dataset = WaveformDataset(seq_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_old)
        val_dataset = WaveformDataset(seq_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_old)
        pass
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
    elif args.network == "CQ_CNNLSTMAttention":
        model = CQ_CNNLSTMAttention(vocab_size=vocab_size, bos_idx=vocab["<BOS>"]).to(device)
    elif args.network == "ResBlock_CodeSequence":
        model = ResBlock_CodeSequence(vocab_size=vocab_size, bos_idx=vocab["<BOS>"]).to(device)

    else:
        raise ValueError(f"指定的网络 {args.network} 无效或不存在")
        # # 使用 DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])  # 定义损失函数和优化器
    class_count = [2588575, 2592940, 1165507, 1164557,  806613,  806634,  807745,  808261,
         269015,  267979,  268122,  268148,  267737,  268920,  268302,  267881,
          90068,   89793,   89697,   89707,   89757,   89452,   89773,   89986,
          89594,   89770,   89959,   89264,   89615,   90275,   89511,   89629,
              1e-5, 1e-5, 1e-5]
    class_weights = 1.0 / torch.tensor(class_count, dtype=torch.float32)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=vocab["<PAD>"])
    # criterion = CosineSimilarityLoss(pad_idx=vocab["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)


    if args.mode == 'train':
        # 训练模型
        train(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, vocab_size)

    elif args.mode == 'test':
        if not args.model_path:
            print("测试模式需要指定模型文件路径，请使用 --model_path 参数。")
            exit(1)
        test(model, seq_val, y_val, criterion, device, args.model_path,vocab)
