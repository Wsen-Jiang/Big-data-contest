import os
import torch
from models.CNN_LSTM_Classifier import CNN_LSTM_Classifier
from dataset import load_data_from_directories
from sklearn.model_selection import train_test_split

def main():
    """
    主函数，用于执行脚本的主要逻辑。

    参数：
        to_pred_dir: 测试集文件夹上层目录路径，不可更改!
        result_save_path: 预测结果文件保存路径，官方已指定为csv格式，不可更改!
    """
    root_dir = 'train_data'
    data_dirs = ['8APSK', '8PSK', '8QAM', '16APSK', '16QAM', '32APSK', '32QAM', 'BPSK', 'MSK', 'QPSK']

    # 读取数据
    sequences, labels = load_data_from_directories(root_dir, data_dirs, "MT")

    # 划分训练集和验证集
    seq_train, seq_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MT_model_path = r"/mnt/data/LWP/Signal-Test/log/models/ModulationType/CNN_LSTM_Classifier/best_model.pth"
    MT_model = CNN_LSTM_Classifier()

    # 加载模型
    if os.path.exists(MT_model_path):
        MT_model.load_state_dict(torch.load(MT_model_path, map_location=device))
        print(f"模型已成功加载: {MT_model_path}")
    else:
        print(f"模型文件不存在: {MT_model_path}")
        return

    # 开始预测
    MT_model.eval()
    correct = 0
    with torch.no_grad():
        for idx, val in enumerate(seq_val):

            sequence = val.unsqueeze(0)  # 增加一个 batch_size 维度，变为 [1, 1727, 2]
            sequence = sequence.permute(0, 2, 1)  # 转置，变为 [1, 2, 1727]

            # 预测调制类型
            logits = MT_model(sequence)
            modulation_type = torch.argmax(logits, dim=1) + 1  # 调制类型预测
            real_label = y_val[idx] + 1  # 实际标签加 1

            print(f'模型预测调制类型：{modulation_type.item()}, 实际真实label:{real_label}')

            if real_label == modulation_type:
                correct += 1

    accuracy = correct / len(seq_val)
    print(f"准确率为：{accuracy:.4f}")

if __name__ == "__main__":
    main()
