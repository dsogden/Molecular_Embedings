import torch
from torch import nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_size: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        return self.dropout(self.embedding(inputs))

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        span = torch.arange(input_dim).unsqueeze(1)
        self.pos = torch.zeros(
            (input_dim, embedding_dim), requires_grad=False
        ).to(device)
        x = torch.arange(embedding_dim // 2).float()
        div_term = torch.exp(x * (-math.log(10000) / embedding_dim))
        self.pos[:, 0::2] = torch.sin(span * div_term)
        self.pos[:, 1::2] = torch.cos(span * div_term)
        self.register_buffer('pe', self.pos)

    def forward(self, inputs):
        return self.dropout(inputs + self.pos)

class Encoder(nn.Module):
    def __init__(
            self, num_embeddings: int, embedding_size: int, max_length: int, num_layers: int
        ):
        super().__init__()
        self.embedding = Embedding(num_embeddings, embedding_size)
        self.positional_encoding = PositionalEncoding(
            max_length, embedding_size
        )
        self.encoder = nn.GRU(
            embedding_size, embedding_size, batch_first=True, num_layers=num_layers
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        embedding = self.embedding(inputs)
        encoded = self.positional_encoding(embedding)
        rnn, hidden = self.encoder(encoded)
        return self.dropout(rnn), hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.permute(0, 2, 1)
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, keys)
        return context

class Decoder(nn.Module):
    def __init__(
            self, num_embeddings: int, embedding_size: int, max_length: int,
            output_size: int, num_layers: int
        ):
        super().__init__()
        self.embedding = Embedding(num_embeddings, embedding_size)
        self.positional_encoding = PositionalEncoding(
            max_length, embedding_size
        )
        self.attention = BahdanauAttention(embedding_size)
        self.decoder = nn.GRU(
            embedding_size, embedding_size, batch_first=True, num_layers=num_layers
        )
        self.fc = nn.Linear(embedding_size, output_size)

    def forward(self, inputs, hidden):
        embedding = self.embedding(inputs)
        encoded = self.positional_encoding(embedding)
        query = hidden.permute(1, 0, 2)
        context = self.attention(query, encoded)
        rnn, _ = self.decoder(encoded + context, hidden)
        output = F.log_softmax(self.fc(rnn), dim=-1)
        B, T, C = output.shape
        return output.view(B * T, C)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, e_inputs, d_inputs):
        rnn, hidden = self.encoder(e_inputs)
        output = self.decoder(d_inputs, hidden)
        return output
