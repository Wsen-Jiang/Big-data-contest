"""
这是一个提交示例Python脚本，供选手参考。
-测评环境为Python3.8
-测评环境中提供了基础的包和框架，具体版本请查看【https://github.com/Datacastle-Algorithm-Department/images/blob/main/doc/py38.md】
-如有测评环境未安装的包，请在requirements.txt里列明, 最好列明版本，例如：numpy==1.23.5
-如不需要安装任何包，请保持requirements.txt文件为空即可，但是提交时一定要有此文件
"""

import os
import pandas as pd
import numpy as np
import torch
from models.CNN_LSTM_Classifier import CNN_LSTM_Classifier
import sys
from utils.dataset import load_data_from_directories, CollateFunction, WaveformDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
def main():
    """
    主函数，用于执行脚本的主要逻辑。

    参数：
        to_pred_dir: 测试集文件夹上层目录路径，不可更改!
        result_save_path: 预测结果文件保存路径，官方已指定为csv格式，不可更改!
    """
    task = 'test'
    # 指定根目录
    root_dir = r'D:\BaiduNetdiskDownload\train_data\train_data'
    # data_dirs = ['8APSK', '8PSK', '8QAM', '16APSK', '16QAM', '32APSK', '32QAM', 'BPSK', 'MSK', 'QPSK']

    # 读取数据
    sequences, labels = load_data_from_directories(root_dir, data_dirs, "MT")

    # 划分训练集和验证集
    seq_train, seq_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )


    """
    读入测试集文件，调用模型进行预测。
    以下预测方式仅为示例参考，实现方式是用循环读入每个文件，依次进行预测，选手可以根据自己模型的情况自行修改
    """
    # 循环测试集文件列表对每个文件进行预测
    correct = 0
    for idx, val in enumerate(seq_val):
        # # 待预测文件路径
        # filepath = os.path.join(testpath, filename)
        #
        # # 读入测试文件，这里的测试文件为无表头的两列信号值
        # df = pd.read_csv(filepath)
        # 尝试提取第一列和第二列作为输入特征
        # 这里为模型加载的路径，可直接使用相对路径
        MT_model_path = r'log/models/ModulationType/CNN_LSTM_Classifier/200_81.25_best_model.pth'
        # SW_model_path = r'./models/SW_best_model.pth'

        MT_model = CNN_LSTM_Classifier()
        # SW_model = CNN_Regressor_LSTM()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        MT_model.load_state_dict(torch.load(MT_model_path, map_location=device))
        # SW_model.load_state_dict(torch.load(SW_model_path, map_location=device))


        # 原始输入 sequence 形状为 [batch_size, sequence_length, num_channels]
        sequence = val.unsqueeze(0)  # 增加一个 batch_size 维度，变为 [1, 1727, 2]
        sequence = sequence.permute(0, 2, 1)  # 转置，变为 [1, 2, 1727]

        # 以下三个需要预测指标分别指定了一个值代替预测值，选手需要根据自己模型进行实际的预测
        # 预测调制类型，数据类型为整型
        logits = MT_model(sequence)
        modulation_type = torch.argmax(logits, dim=1)+1
        real_label = y_val[idx]+1
        print(f'模型预测调制类型：{modulation_type.item()},      实际真实label:{real_label}')
        if(real_label == modulation_type):
            correct += 1

    print("准确率为：{}".format(correct/len(seq_val)))
    # 将预测结果保存到result_save_path,保存方式可修改，但是注意保存路径不可更改！！！
    # with open(result_save_path, 'w') as f:
    #     f.write('\n'.join(result))

    # 如果result为已经预测好的DataFrame数据，则可以直接使用pd.to_csv()的方式进行保存
    # result.to_csv(result_save_path, index=None)

if __name__ == "__main__":
    main()
