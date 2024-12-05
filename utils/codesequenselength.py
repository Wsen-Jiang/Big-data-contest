# import os
# import pandas as pd
# import numpy as np
#
# root_dir = './Dataset'
# data_dirs = ['1', '2', '3', '4']
#
# for dir_name in data_dirs:
#     dir_path = os.path.join(root_dir, dir_name)
#     for file_name in os.listdir(dir_path):
#         if file_name.endswith('.csv'):
#             file_path = os.path.join(dir_path, file_name)
#             try:
#                 # 读取CSV文件，无标题行
#                 data = pd.read_excel(file_path, engine='xlrd',header=None)
#
#                 # 删除第3列中的 NaN 值以计算码序列长度
#                 data_nonan = data.dropna(subset=[2])
#                 CodeSequence = data_nonan.iloc[:, 2].values.astype(np.int32)
#                 code_sequence_length = len(CodeSequence)
#                 print(f"文件 {file_name} 的码序列长度: {code_sequence_length}")
#
#                 # 在原始数据上添加新的列，第6列为码序列长度
#                 data[5] = code_sequence_length
#
#                 # 保存数据，不修改原始行数
#                 data.to_csv(file_path, index=False, header=False)
#
#             except IndexError:
#                 print(f"文件 {file_name} 的列数不足，跳过该文件")
#             except Exception as e:
#                 print(f"处理文件 {file_name} 时出现错误: {e}")
import os
import pandas as pd
import numpy as np

root_dir = './Dataset'
data_dirs = ['1', '2', '3', '4']

for dir_name in data_dirs:
    dir_path = os.path.join(root_dir, dir_name)
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(dir_path, file_name)
            try:
                # 读取之前已保存的 CSV 文件
                data = pd.read_csv(file_path, header=None)

                # 保留第 0 行的第 5 列，删除其他行中第 5 列的值
                data.iloc[1:, 5] = np.nan  # 将其他行第 5 列的值设置为 NaN

                # 如果希望删除所有有 NaN 值的行，可以使用 dropna，但这里只保留第 0 行第 5 列的值
                # data = data.dropna(subset=[5])

                # 将修改后的数据写回 CSV 文件
                data.to_csv(file_path, index=False, header=False, encoding='utf-8-sig')

                print(f"文件 {file_name} 已处理完成")

            except IndexError:
                print(f"文件 {file_name} 的列数不足，跳过该文件")
            except Exception as e:
                print(f"处理文件 {file_name} 时出现错误: {e}")
