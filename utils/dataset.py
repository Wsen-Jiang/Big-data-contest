from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from torch.nn.utils.rnn import pad_sequence
import scipy.signal as signal
import torch.nn.functional as F

# 自定义collate_fn类，用于处理变长序列
# 在处理变长序列时，默认的 DataLoader 无法将不同长度的序列组合成一个批次，因此需要自定义 collate_fn 函数来实现填充操作。

def lowpass_filter_iq(iq_tensor, cutoff_freq, fs, filter_order=4):
    """
    对输入的 IQ 信号进行低通滤波
    参数:
        iq_tensor: torch.Tensor, 输入信号，维度 (L, 2)
        cutoff_freq: float, 低通滤波器的截止频率
        fs: float, 信号的采样频率（20 MHz）
        filter_order: int, 滤波器的阶数
    返回:
        filtered_tensor: torch.Tensor, 滤波后的 IQ 信号
    """
    # 检查输入 Tensor 格式
    assert iq_tensor.shape[1] == 2, "输入的 Tensor 维度必须为 (L, 2)"

    # 将 Tensor 转换为 NumPy 数组
    iq_numpy = iq_tensor.numpy()

    # 设计低通滤波器
    nyquist = fs / 2  # 奈奎斯特频率
    normal_cutoff = cutoff_freq / nyquist  # 归一化截止频率
    b, a = signal.butter(filter_order, normal_cutoff, btype='low')

    # 对 I 和 Q 分量分别进行滤波
    i_filtered = signal.filtfilt(b, a, iq_numpy[:, 0])  # I 分量
    q_filtered = signal.filtfilt(b, a, iq_numpy[:, 1])  # Q 分量

    # 将结果组合成新的 Tensor
    filtered_numpy = np.column_stack((i_filtered, q_filtered))
    filtered_tensor = torch.tensor(filtered_numpy, dtype=torch.float32)

    return filtered_tensor


def add_awgn(signal, SNR):
    noise = torch.randn_like(signal)
    signal_power = torch.mean((signal - torch.mean(signal)) ** 2)
    noise_variance = signal_power / (10 ** (SNR / 10))
    noise *= (torch.sqrt(noise_variance) / torch.std(noise,unbiased=False))
    signal_noise = signal + noise
    return signal_noise

class CollateFunction:
    def __init__(self,train_mode = None,vocab = None):
        self.train_mode = train_mode
        self.vocab = vocab
    def __call__(self,batch):
        sequences = [add_awgn(item[0], 20) for item in batch]  # IQ 序列 [L,2]

        # fs = 20e6  # 采样频率：20 MHz
        # cutoff_freq = 5e6  # 截止频率：5 MHz
        # sequences = [lowpass_filter_iq(item, cutoff_freq, fs) for item in sequences] # 低通滤波

        # 获取序列长度
        seq_lengths = torch.tensor([256 for seq in sequences], dtype=torch.long)
        
        # 填充全序列
        # padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        

        # 经验：256前向固定长度截取，性价比最高
        ml = 256
        padded_sequences = []
        for seq in sequences:
            if len(seq) >= ml:
                # 随机选择起始索引
                start_idx = torch.randint(0, len(seq) - ml + 1, (1,)).item()
                # 截取长度为256的子序列
                truncated_seq = seq[start_idx : start_idx + ml]
                padded_sequences.append(truncated_seq)
            else:
                 padded_sequences.append(F.pad(seq, (0, 0, 0, ml - seq.size(0)), value=0.0))
        padded_sequences = torch.stack(padded_sequences)

        padded_sequences = padded_sequences.permute(0, 2, 1)
        
        if self.train_mode == "MT":         # 分类任务
            labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        elif self.train_mode == "SW":       # 回归任务
            labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
        elif self.train_mode == "CQ":       # 码序列生成
            labels = [item[1].clone().detach().long() for item in batch] #每一个码序列是一个tensor
            label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
            # 填充标签，使用PAD作为填充值
            labels = pad_sequence(labels, batch_first=True, padding_value=self.vocab["<PAD>"])
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
            if train_mode == "MT":
                labels = processed_data['mt_labels']
            elif train_mode == "SW":
                labels = processed_data['sw_labels']
            elif train_mode == "CQ":
                labels = processed_data['cq_labels']
            else:
                raise ValueError(f"无效的 train_model 参数: {train_mode}")
    else:
        sequences = []
        mt_labels = []
        sw_labels = []
        cq_labels = []

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

                        # 归一化
                        # sequence = normalization(sequence)

                        # 添加到列表
                        sequences.append(sequence)

                        # 调制类型
                        ModulationType = int(data.iloc[0, 3]) - 1 # 获取第四列的第一个元素，0 表示第一行，3 表示第四列
                        # 码元宽度
                        SymbolWidth = round(float(data.iloc[0, 4]), 2)
                        # 码序列
                        data = data.dropna(subset=[2]) # 删除码序列中的 NaN 值
                        CodeSequence = data.iloc[:,2].values.astype(np.int32)

                        mt_labels.append(torch.tensor(ModulationType))
                        sw_labels.append(torch.tensor(SymbolWidth))
                        cq_labels.append(torch.tensor(CodeSequence))

                    except IndexError:
                        print(f"文件 {file_name} 的列数不足，跳过该文件")

        os.mkdir(os.path.join(root_dir, 'cache'))
        with open(os.path.join(root_dir, 'cache', 'processed_data.pkl'), 'wb') as f:
            processed_data = {}
            processed_data['sequences'] = sequences
            processed_data['mt_labels'] = mt_labels
            processed_data['sw_labels'] = sw_labels
            processed_data['cq_labels'] = cq_labels
            pkl.dump(processed_data, f)
        if train_mode == "MT":
            labels = mt_labels
        elif train_mode == "SW":
            labels = sw_labels
        elif train_mode == "CQ":
            labels = cq_labels
        else:
            raise ValueError(f"无效的 train_model 参数: {train_mode}")
    return sequences, labels