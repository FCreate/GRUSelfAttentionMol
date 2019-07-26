import torch
import torch.nn as nn

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
    def __init__(self, rnn_hidden, out_hid, r):
        super(BahdanauSA, self).__init__()
        self.out_hid = out_hid
        self.rnn_hidden = rnn_hidden
        self.r = r
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

class LinearLayer(nn.Module):
    def __init__(self, size_in, size_out, p=0.5, is_bn=True, activation=True):
        super(LinearLayer, self).__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.is_bn = is_bn
        self.activation = activation
        self.p = p
        self.fc = nn.Linear(self.size_in, self.size_out)
        self.dropout = nn.Dropout(self.p)
        self.elu = nn.ELU()
        if self.is_bn:
            self.batchnorm = nn.BatchNorm1d(size_out)
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.01)

    def forward(self, x):
        output = self.fc(x)
        if self.is_bn:
            output = self.batchnorm(output)
        if self.activation:
            output = self.elu(output)
        output = self.dropout(output)
        return output


class Decoder(nn.Module):
    def __init__(self,in_shape, out_shape):
        super(Decoder, self).__init__()
        self.linear1 = LinearLayer(in_shape, 128, 0.0, is_bn = False)
        #self.linear1_5 = LinearLayer(1024, 512, 0.5)
        # self.linear1_5 = LinearLayer(4096, 2048, 0.5)
        # self.linear1_7 = LinearLayer(2048, 1024, 0.5)
        # self.linear1_8 = LinearLayer(4096, 512, 0.5)
        # self.linear1_5 = LinearLayer(1024, 512, 0.5)
        #self.linear2 = LinearLayer(512, 256, 0.5)
        self.linear3 = LinearLayer(128, 64, 0.0, is_bn = False)
        #self.linear4 = LinearLayer(128, 64, 0.5)
        #self.linear5 = LinearLayer(64, 32, 0.25)
        self.linear6 = LinearLayer(64, 29, 0.0, is_bn = False)
        self.linear7 = LinearLayer(29, out_shape, 0.0, is_bn=False, activation=False)
        self.layers = [self.linear1, self.linear3, self.linear6,self.linear7]

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

class Model(nn.Module):
    def __init__(self, args, ntokens, number_of_words, n_endpoints):
        super(Model, self).__init__()
        self.n_endpoints = n_endpoints
        self.number_of_words = number_of_words

        self.embedding = nn.Embedding(ntokens, args.emsize)
        self.encoder = Encoder(args.emsize, args.hidden, nlayers=args.nlayers, dropout=args.drop, bidirectional=args.bi)
        self.attention_dim = args.hidden if not args.bi else 2 * args.hidden
        self.attention = BahdanauSA(self.attention_dim, args.hid_sa_val, args.r)
        self.flatten = Flatten()
        self.decoder = Decoder(self.attention.r*self.number_of_words, self.n_endpoints)


    def forward(self, input):
        outputs, _ = self.encoder(self.embedding(input)) #outputs - hidden values from rnn [batch_size*embed_size*(bi*hidden_layer)],[embed_size*(bi*hidden_layer)]
        out = self.flatten(self.attention(outputs))
        predictions = self.decoder(out)
        return predictions


    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.attention.to(*args, **kwargs)
        return self
