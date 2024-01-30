import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.recurrent = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers, batch_first=True)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        outputs, (hidden, cell) = self.recurrent(embedded)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, n_layers):
      super(Decoder, self).__init__()
      self.input_size = input_size
      self.embedding_size = embedding_size
      self.hidden_size = hidden_size
      self.output_size = output_size
      self.n_layers = n_layers
    
      self.embedding = nn.Embedding(self.input_size, self.embedding_size)
      self.recurrent = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers, batch_first=True)
      self.logits = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, hidden, labels):
      batch_size = 32
      decoder_input = torch.zeros(batch_size, 1, dtype=torch.long)
      decoder_outputs = []
    
      for i in range(26):
          embedded = F.relu(self.embedding(decoder_input))
          outputs, hidden = self.recurrent(embedded, hidden)
          logits = self.logits(outputs)
          decoder_outputs.append(logits)
    
          if labels is not None:
              decoder_input = labels[:, i].unsqueeze(1)
          else:
              _, topi = logits.topk(1)
              decoder_input = topi.squeeze(-1).detach()
    
      decoder_outputs = torch.cat(decoder_outputs, dim=1)
      decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
      return decoder_outputs
