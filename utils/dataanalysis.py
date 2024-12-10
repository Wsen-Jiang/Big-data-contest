import os
import sys
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

data_pth = '/mnt/data/LXP/data/train_data'
modes = os.listdir(data_pth)
print(modes)
cntr = Counter()
cnt = 0
for mode in modes:
    file_lists = os.listdir(os.path.join(data_pth, mode))
    print(mode, ': ', len(file_lists))
    for file in file_lists:
        file_path = os.path.join(data_pth, mode, file)
        try:
            data = pd.read_csv(file_path, header=None)

            # 调制类型
            ModulationType = int(data.iloc[0, 3]) - 1  # 获取第四列的第一个元素，0 表示第一行，3 表示第四列
            # 码元宽度
            SymbolWidth = round(float(data.iloc[0, 4]), 2)
            # 码序列
            data = data.dropna(subset=[2])  # 删除码序列中的 NaN 值
            CodeSequence = data.iloc[:, 2].values.astype(np.int32)
            cntr[len(CodeSequence)] += 1
        except IndexError:
            print(f"文件 {file_path} 的列数不足，跳过该文件")
        except UnicodeDecodeError:
            print(f"文件 {file_path} 的编码错误")

cc_keys = list(cntr.keys())
cc_keys.sort()
x_values, y_values = [], []
for k in cc_keys:
    print(k, cntr[k])
    x_values.append(k)
    y_values.append(cntr[k])
plt.bar(x_values, y_values)
plt.xlabel('Code Sequence length')
plt.ylabel('Count')
plt.show()
sys.exit()