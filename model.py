import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Encoder(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, nlayers=1, dropout=0.,
               bidirectional=True):
    super(Encoder, self).__init__()
    self.bidirectional = bidirectional
    self.rnn = nn.GRU(embedding_dim, hidden_dim, nlayers,
                        dropout=dropout, bidirectional=bidirectional)

  def forward(self, input, hidden=None):
    val  = self.rnn(input, hidden)
    return val

class BahdanauSA(nn.Module):
    def __init__(self, rnn_hidden, out_hid, r, device):
      super(BahdanauSA, self).__init__()
      self.out_hid = out_hid
      self.rnn_hidden = rnn_hidden
      self.r = r
      self.device = device
      self.w1 = torch.randn((self.out_hid,self.rnn_hidden), dtype = torch.float32, requires_grad=True)
      self.tanh = nn.Tanh()
      self.w2 = torch.randn((self.r, self.out_hid), dtype=torch.float32, requires_grad=True)
      self.softmax = nn.Softmax(dim = 2)
    def forward(self, H):

      out= torch.matmul(self.w1, torch.transpose(H, 1,2))
      out = self.tanh(out)
      out = torch.matmul(self.w2, out)
      out = self.softmax(out)
      return out

    def to(self, *args, **kwargs):
      self = super().to(*args, **kwargs)
      self.w1 = self.w1.to(*args, **kwargs)
      self.w2 = self.w2.to(*args, **kwargs)
      return self


class Flatten(nn.Module):
  def forward(self, input):
    return input.view(input.size(0), -1)


class Classifier(nn.Module):
  def __init__(self, embedding, encoder,attention, n_endpoints):
    super(Classifier, self).__init__()
    self.embedding = embedding
    self.encoder = encoder
    self.attention = attention
    self.n_endpoints = n_endpoints
    self.decoder = nn.Linear(self.attention.r*self.attention.out_hid, n_endpoints)
    self.flatten = Flatten()


  def forward(self, input):
    outputs, _ = self.encoder(self.embedding(input))#outputs - hidden values from rnn [batch_size*embed_size*(bi*hidden_layer)],[embed_size*(bi*hidden_layer)]
    out = self.flatten(self.attention(outputs))
    predictions = self.decoder(out)
    return predictions


  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.attention.to(*args, **kwargs)
    return self
