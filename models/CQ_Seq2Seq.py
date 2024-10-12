import torch.nn as nn
import torch

# input_dim = 2  # IQ流的输入维度
# output_dim = 16  # 码序列的词汇表大小
# emb_dim = 64  # 嵌入维度
# hidden_dim = 128  # 隐藏层维度
# n_lays = 2  # LSTM层数
# dropout = 0.5  # Dropout概率
class Encoder(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim = 128, n_layers = 2, dropout = 0.5):
        super(Encoder, self).__init__()
        # batch_first = True 表示输入和输出张量的第一个维度是批次大小
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

    def forward(self, seq, seq_lengths):
        seq_lengths = seq_lengths.cpu().type(torch.int64)
        # 使用pack_padded_sequence函数将序列数据打包，以便LSTM层可以忽略填充的部分。
        packed_embedded = nn.utils.rnn.pack_padded_sequence(seq, seq_lengths, batch_first=True, enforce_sorted=False)
        # (batch_size, seq_length, input_dim) -> (batch_size, seq_length, hidden_dim)
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim = 16, emb_dim = 64, hidden_dim = 128, n_layers = 2, dropout = 0.5):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)  # 将输入形状变为 (batch_size, 1)
        embedded = self.dropout(self.embedding(input))  # 嵌入后 (batch_size, 1, emb_dim)

        # 这里 embedded 应该是 3D 张量
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # 输入应为 (batch_size, 1, emb_dim)

        prediction = self.fc_out(output)  # (batch_size, 1, output_dim)
        return prediction, hidden, cell

class CQ_Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(CQ_Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, seq, seq_lengths, label, label_lengths):
        hidden, cell = self.encoder(seq, seq_lengths)
        output = torch.zeros(label.size(0), label.size(1), self.decoder.output_dim).to(seq.device)

        input = label[:, 0]  # 首个输入为起始符
        for t in range(1, label_lengths.max().item() + 1):  # 控制生成的长度
            output_t, hidden, cell = self.decoder(input, hidden, cell)
            output[:, t - 1, :] = output_t.squeeze(1)  # 这里应保持 output_t 的形状正确

            # 选择解码器输出中概率最高的元素作为下一个时间步的输入
            input = output_t.argmax(dim=2).squeeze(1)

        return output