import torch.nn as nn

# class CQ_LSTM(nn.Module):
#     def __init__(self, input_size=2, hidden_size=128, output_size=16, num_layers=1):
#         super(CQ_LSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x, lengths):
#         # 使用 pack_padded_sequence 处理变长序列
#         packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
#         packed_output, (hn, cn) = self.lstm(packed_input)
#         # 解包
#         output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
#         # 取最后一个有效时间步的输出
#         idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(output.size(0), 1, output.size(2))
#         last_output = output.gather(1, idx).squeeze(1)
#         # 全连接层
#         out = self.fc(last_output)
#         return out
class CQ_LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, embedding_dim=128):
        super(CQ_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 输出层，将隐藏状态映射到嵌入空间
        self.fc = nn.Linear(hidden_size, embedding_dim)

    def forward(self, x, lengths):
        # 打包输入
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        # 解包输出
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # 映射到嵌入空间
        output = self.fc(output)
        return output