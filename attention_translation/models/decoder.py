import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, attention, dropout_p=0.1):
        super(Decoder, self).__init__()
        # [batch_size, output_dim] -> [batch_size, output_dim, hidden_dim]
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.attention = attention
        self.rnn = nn.GRU(hidden_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(x))
        # embedded = self.embedding[x].unsqueeze(1)  # Shape: (batch_size, 1, output_dim)

        context, attention_weights = self.attention(hidden.transpose(0, 1), encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)

        output, hidden = self.rnn(rnn_input, hidden)

        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, attention_weights
