# Big-data-contest

## task 1 
### experiment result:

| Model           | Learning Rate | Batch Size | Accuracy(batch) | Accurcay | ext         |
|-----------------|---------------|------------|----------|-------------|-----------------|
| CNN*2_Transform | 0.005         | 512        | 60.24    |     | overfitting |
| CNN*3_LSTM | 0.005 | 1024 |  | 58.49 | overfitting |
| CNN*3_LSTM      | 0.003         | 2048       | 60.03    |     | overfitting |
| CNN*3_LSTM      | 0.005         | 2048       | 60.81    |     | overfitting |
| CNN*3_GRU       | 0.005         | 4096       | 43.32    |     | overfitting |
| CNN*2_LSTM      | 0.005         | 2048       | 62.63    | 55.57 | overfitting |
| CNN*3_LSTM*2            | 0.005         | 2048       | 27.24    |     | 填充循环序列      |
| CNN*3_LSTM*2_可分离卷积_残差网络 | 0.005         | 2048       | 61.2     |      | overfitting |

## task 2 
### experiment result:
| Model           | Learning Rate | Batch Size | Score | ext |
|-----------------|---------------|------------|-------|-----|
| CNN*3_Transform | 0.005         | ***        | ***   | ***    |
| CNN*3_LSTM*2    | 0.005         | 2048  | 94.22 | 表现最好   |
| CNN*3_LSTM*2    | 0.005         | 2048       | 91.83 | 填充循环序列 |


## task 3 
### experiment result:
| Model           | Learning Rate | Batch Size | Score | ext |
|-----------------|---------------|------------|-------|-----|
| CNN*3_Transform | 0.005         | ***        | ***   | *** |

## Leaderboard
![Image text](https://github.com/WenSen-Jiang/Big-data-contest/blob/main/fig/leaderboard.png)
