import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models.CNNRegressor import CNNRegressor
from models.ResidualRegressor import ResidualRegressor
from dataset import load_data_from_directories, WaveformDataset, CollateFunction

from sklearn.metrics import mean_squared_error, mean_absolute_error

# symbol_width_to_index = {
#     0.25: 0, 0.3: 1, 0.35: 2, 0.4: 3, 0.45: 4,
#     0.5: 5, 0.55: 6, 0.6: 7, 0.65: 8, 0.7: 9,
#     0.75: 10, 0.8: 11, 0.85: 12, 0.9: 13, 0.95: 14, 1.0: 15
# }
#
# # 交换键值对
# index_to_symbol_width = {v: k for k, v in symbol_width_to_index.items()}

if __name__ == "__main__":
    # 指定根目录
    root_dir = 'Dataset'
    data_dirs = ['1','2','3','4']

    # 读取数据
    sequences, labels = load_data_from_directories(root_dir, data_dirs, "SW")

    seq_train, seq_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    collate_fn = CollateFunction(train_mode="SW")
    val_dataset = WaveformDataset(seq_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

    model = ResidualRegressor()
    model_path = r"./log/models/SymbolWidth/ResidualRegressor/best_model.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型已成功加载: {model_path}")

    model.eval()
    # correct = 0
    total_relative_error = 0
    total_samples = 0
    with torch.no_grad():
        for batch_X, lengths, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            lengths = lengths.to(device)

            batch_X = batch_X.permute(0, 2, 1)  # [batch_size, channels, seq_len]

            outputs = model(batch_X)
            for i in range(len(outputs)):
                # 将标签从索引转换为符号宽度
                label = batch_y[i].item()

                # 获取模型输出并进行修正
                output = outputs[i].item()
                if output < 0.2:
                    output = 0.2
                elif output > 1:
                    output = 1

                #计算相对误差
                relative_error = abs(output - label) / label
                print(f"标签: {label}, 输出: {output}, 相对误差：{relative_error}")
    #             # 对输出进行规整
    #             output_round = round(output * 20) / 20
    #
    #             # 比较输出和标签
    #             if output_round == label:
    #                 correct += 1
                total_relative_error += relative_error
                total_samples += 1
    #             print(f"标签: {label}, 输出: {output}, 输出规整后: {output_round}")
    print(f"平均相对误差: {total_relative_error / total_samples}")
    # print(f"总样本数: {total_samples}")
    # print(f"正确数: {correct}")
    # print(f"准确率: {correct / total_samples}")
