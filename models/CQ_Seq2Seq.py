import torch.nn as nn
import torch

# input_dim = 2  # IQ流的输入维度
# output_dim = 35  # 码序列的词汇表大小
# emb_dim = 64  # 嵌入维度
# hidden_dim = 64  # 隐藏层维度
# n_lays = 2  # LSTM层数
# dropout = 0.5  # Dropout概率
class Encoder(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim = 64, n_layers = 2, dropout = 0.5):
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
    def __init__(self, output_dim = 35, emb_dim = 64, hidden_dim = 64, n_layers = 2, dropout = 0.5):
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
    def __init__(self, encoder, decoder,vocab):
        super(CQ_Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab
        self.start_token_idx = vocab["<BOS>"]  # 起始符
        self.end_token_idx = vocab["<EOS>"]  # 结束符
        self.pad_token_idx = vocab["<PAD>"]  # 填充符
        self.max_decoding_steps = 400  # 解码的最大步数

    def forward(self, seq, seq_lengths, label, label_lengths,use_teacher_forcing=True):
        hidden, cell = self.encoder(seq, seq_lengths)
        # print(hidden.shape)
        batch_size = seq.size(0)
        output_dim = self.decoder.output_dim

        # max_len = label_lengths.max().item()

        output = torch.full(
            (batch_size, self.max_decoding_steps, output_dim),
            fill_value=self.pad_token_idx,
            device=seq.device,
            dtype=torch.float
        )

        teacher_forcing_ratio = 0.5  # 50% 概率使用教师强制
        # 用于记录每个序列是否已经完成生成
        completed = torch.zeros(batch_size, dtype=torch.bool, device=seq.device)
        # 使用起始符初始化解码器输入
        input = torch.full((batch_size,), self.start_token_idx, dtype=torch.long, device=seq.device)

        for t in range(self.max_decoding_steps):  # 控制生成的长度
            if not completed.all():
                output_t, hidden, cell = self.decoder(input, hidden, cell)
                
                # print(output_t.squeeze(1))
                # output[:, t, :] = output_t.squeeze(1)  # 保持 output_t 的形状正确
                # 获取当前步的预测结果
                top = output_t.argmax(dim=-1).squeeze(1)
                # 如果完成了输出EOS，则更新completed标记
                completed = completed | (top == self.end_token_idx)
                if t == 0:
                    output[:, t, :] = output_t.squeeze(1)
                else:
                    # 只更新未完成的序列
                    not_completed = ~completed
                    # print(output[not_completed, t, :].shape)
                    # print(output_t.squeeze(1)[not_completed].shape)
                    output[not_completed, t, :] = output_t.squeeze(1)[not_completed]
                # 更新下一个输入，仅对未完成的序列更新
                if use_teacher_forcing and label is not None and  0< t < label.size(1):
                    teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                    next_input = label[:, t] if teacher_force else top
                else:
                    next_input = top
                input = torch.where(completed, torch.full_like(input, self.pad_token_idx), next_input)

            else:
                break
        # # 使用 softmax 得到概率分布 (512, 398, 16)
        # probabilities = torch.softmax(output, dim=-1)  # 在最后一个维度做 softmax
        #
        # # 根据概率分布计算加权平均的嵌入表示
        # predicted_sequences = probabilities @ torch.arange(self.decoder.output_dim, device=seq.device).float()
        # # predicted_sequences = torch.round(predicted_sequences).to(torch.int64)  # 四舍五入取整
        #
        # return predicted_sequences  # argmax、round\torch.int64无法微分，导致梯度无法传递
        return output