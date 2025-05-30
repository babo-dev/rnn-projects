import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, query, keys):
        """
        Args:
            query: Decoder hidden state (batch_size, 1, hidden_dim).
            keys: Encoder annotations (batch_size, seq_len, hidden_dim).

        Returns:
            context_vector: Weighted sum of encoder annotations (batch_size, hidden_dim).
            attention_weights: Attention weights (batch_size, seq_len).
        """
        # Compute alignment scores (dot product)
        scores = torch.bmm(keys, query.transpose(1, 2)).squeeze(2)  # Shape: (batch_size, seq_len)

        # Compute attention weights using softmax
        attention_weights = F.softmax(scores, dim=1)  # Shape: (batch_size, seq_len)

        # Compute context vector as a weighted sum of encoder annotations
        context_vector = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)  # Shape: (batch_size, hidden_dim)

        return context_vector, attention_weights


class GeneralAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(GeneralAttention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)  # Learnable weight matrix

    def forward(self, query, keys):
        """
        Args:
            query: Decoder hidden state (batch_size, 1, hidden_dim).
            keys: Encoder annotations (batch_size, seq_len, hidden_dim).

        Returns:
            context_vector: Weighted sum of encoder annotations (batch_size, hidden_dim).
            attention_weights: Attention weights (batch_size, seq_len).
        """
        # Transform encoder annotations using the learnable weight matrix
        transformed_keys = self.W(keys)  # Shape: (batch_size, seq_len, hidden_dim)

        # Compute alignment scores (dot product)
        scores = torch.bmm(transformed_keys, query.transpose(1, 2)).squeeze(2)  # Shape: (batch_size, seq_len)

        # Compute attention weights using softmax
        attention_weights = F.softmax(scores, dim=1)  # Shape: (batch_size, seq_len)

        # Compute context vector as a weighted sum of encoder annotations
        context_vector = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)  # Shape: (batch_size, hidden_dim)

        return context_vector, attention_weights


class ConcatAttention(nn.Module):  # Also known as Bahdanau attention; Additive attention
    def __init__(self, hidden_dim):
        super(ConcatAttention, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # For encoder annotations
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # For decoder hidden state
        self.V = nn.Linear(hidden_dim, 1, bias=False)           # For scoring

    def forward(self, query, keys, mask=None):
        """
        Args:
            query: Decoder hidden state (batch_size, 1, hidden_dim).
            keys: Encoder annotations (batch_size, seq_len, hidden_dim).

        Returns:
            context_vector: Weighted sum of encoder annotations (batch_size, hidden_dim).
            attention_weights: Attention weights (batch_size, seq_len).
        """
        # Expand query to match the sequence length of keys
        query = query.expand(-1, keys.size(1), -1)  # Shape: (batch_size, seq_len, hidden_dim)

        # Compute alignment scores using a feedforward network
        scores = self.V(torch.tanh(self.W1(keys) + self.W2(query)))  # Shape: (batch_size, seq_len, 1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # Mask padding tokens
        scores = scores.squeeze(2)  # Shape: (batch_size, seq_len)

        # Compute attention weights using softmax
        attention_weights = F.softmax(scores, dim=1)  # Shape: (batch_size, seq_len)

        # Compute context vector as a weighted sum of encoder annotations
        context_vector = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)  # Shape: (batch_size, hidden_dim)

        return context_vector, attention_weights
