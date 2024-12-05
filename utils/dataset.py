from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from torch.nn.utils.rnn import pad_sequence
# 自定义collate_fn类，用于处理变长序列
# 在处理变长序列时，默认的 DataLoader 无法将不同长度的序列组合成一个批次，因此需要自定义 collate_fn 函数来实现填充操作。
class CollateFunction:
    def __init__(self,train_mode = None):
        self.train_mode = train_mode
    def __call__(self,batch):
        sequences = [item[0] for item in batch]  # IQ 序列
        # 获取序列长度
        seq_lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        # 填充序列
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)

        if self.train_mode == "MT":         # 分类任务
            labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        elif self.train_mode == "SW":       # 回归任务
            labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
        elif self.train_mode == "CQ":       # 码序列生成
            labels = [torch.tensor(item[1], dtype=torch.long) for item in batch] #每一个码序列是一个tensor
            label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
            # 填充标签，使用 -1 作为填充值
            labels = pad_sequence(labels, batch_first=True, padding_value=-1)
            return padded_sequences, seq_lengths, labels, label_lengths
        else:
            raise ValueError(f"无效的 train_mode 参数: {self.train_mode}")
        return padded_sequences, seq_lengths, labels


# 定义自定义数据集类
class WaveformDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences  # 序列列表，每个元素是一个Tensor
        self.labels = labels  # 标签列表

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]  # 获取序列
        label = self.labels[idx]  # 获取标签
        return seq, label


def normalization(input_tensor):
    input_tensor = input_tensor.permute(1,0)
    amplitude = torch.sqrt(input_tensor[0]**2 + input_tensor[1]**2)
    amplitude_max = torch.max(amplitude)
    # 实部和虚部归一化
    normalized_real = input_tensor[0] / amplitude_max
    normalized_imag = input_tensor[1] / amplitude_max
    # 组合归一化结果
    normalized_tensor = torch.stack([normalized_real, normalized_imag]) # [2, seq_len]
    normalized_tensor = normalized_tensor.permute(1,0) # [seq_len, 2]
    return normalized_tensor

# 数据读取函数
def load_data_from_directories(root_dir, data_dirs,train_mode):
    if os.path.exists(os.path.join(root_dir, 'cache', 'processed_data.pkl')):
        with open(os.path.join(root_dir, 'cache', 'processed_data.pkl'), 'rb') as f:
            processed_data = pkl.load(f)
            sequences = processed_data['sequences']
            labels = processed_data['labels']
    else:
        sequences = []
        labels = []

        # 遍历每个目录
        for dir_name in data_dirs:
            dir_path = os.path.join(root_dir, dir_name)
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(dir_path, file_name)
                    try:
                        # 读取CSV文件，无标题行
                        # data = pd.read_excel(file_path, engine='xlrd', header=None)
                        data = pd.read_csv(file_path, header=None)

                        # 尝试提取第一列和第二列作为输入特征
                        sequence = data.iloc[:, [0, 1]].values.astype(np.float32)
                        sequence = torch.tensor(sequence)
                        # 添加到列表
                        sequences.append(sequence)

                        # 调制类型
                        ModulationType = int(data.iloc[0, 3]) - 1 # 获取第四列的第一个元素，0 表示第一行，3 表示第四列
                        # 码元宽度
                        SymbolWidth = round(float(data.iloc[0, 4]), 2)
                        # 码序列
                        data = data.dropna(subset=[2]) # 删除码序列中的 NaN 值
                        CodeSequence = data.iloc[:,2].values.astype(np.int32)

                        if train_mode == "MT":
                            label = ModulationType
                        elif train_mode == "SW":
                            label = SymbolWidth
                        elif train_mode == "CQ":
                            label = CodeSequence
                        else:
                            raise ValueError(f"无效的 train_model 参数: {train_mode}")
                        label = torch.tensor(label)
                        labels.append(label)

                    except IndexError:
                        print(f"文件 {file_name} 的列数不足，跳过该文件")

        os.mkdir(os.path.join(root_dir, 'cache'))
        with open(os.path.join(root_dir, 'cache', 'processed_data.pkl'), 'wb') as f:
            processed_data = {}
            processed_data['sequences'] = sequences
            processed_data['labels'] = labels
            pkl.dump(processed_data, f)
    return sequences, labels