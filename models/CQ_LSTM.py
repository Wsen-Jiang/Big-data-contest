import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, embedding_dim=128):
        super(LSTMModel, self).__init__()
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