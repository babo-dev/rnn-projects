import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()
        # [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        # hidden = hidden[-1].unsqueeze(0)
        return outputs, hidden


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)  # [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)  
        # [batch_size, seq_len, hidden_size] -> ( [batch_size, seq_len, hidden_size], [num_layer, batch_size, hidden_size] )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
