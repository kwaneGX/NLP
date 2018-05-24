import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import torch.nn.functional as F

import const


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input):
        mu = torch.mean(input, dim=-1, keepdim=True)
        sigma = torch.std(input, dim=-1, keepdim=True).clamp(min=self.eps)
        output = (input - mu) / sigma
        return output * self.weight.expand_as(output) + self.bias.expand_as(output)

class LSTM_Text(nn.Module):

    def __init__(self,vocab_size,batch_size,embed_dim,label_size):
        super(LSTM_Text,self).__init__()
        self.vocab_size=vocab_size
        self.embed_dim=embed_dim
        self.hidden_size=200
        self.lstm_layers=1
        self.dropout=0.5
        self.batch_size=batch_size
        self.bidirectional=1
        self.label_size=label_size
        #self.num_directions = 2 if self.bidirectional else 1
        self.num_directions = 2 if 1 else 1

        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim,
                                padding_idx=const.PAD)
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.lstm_layers,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)
        self.ln = LayerNorm(self.hidden_size*self.num_directions)
        self.logistic = nn.Linear(self.hidden_size*self.num_directions,
                                self.label_size)

        self._init_weights()

    def _init_weights(self, scope=1.):
        self.lookup_table.weight.data.uniform_(-scope, scope)
        #self.lookup_table.weight.data.copy_(torch.from_numpy(pretrained_wordvec))
        self.logistic.weight.data.uniform_(-scope, scope)
        self.logistic.bias.data.fill_(0)

    def init_hidden(self):
        num_layers = self.lstm_layers*self.num_directions

        weight = next(self.parameters()).data
        return (Variable(weight.new(num_layers, self.batch_size, self.hidden_size).zero_()),Variable(weight.new(num_layers, self.batch_size, self.hidden_size).zero_()))

    def forward(self, input,mask, hidden):
        
        encode = self.lookup_table(input)
        lstm_out, hidden = self.lstm(encode.transpose(0, 1), hidden)
        #print lstm_out.size(),mask[:, :, None].size(),mask,encode.transpose(0, 1).size()
        #print mask[:, :, None].transpose(0, 1)
        
        output = self.ln(lstm_out)
        final_h = torch.mul(output, mask[:, :, None].transpose(0, 1).type(torch.cuda.FloatTensor))
        final_h=torch.sum(final_h.transpose(0, 1), 1)
        #output = self.ln(lstm_out)[-1]
        #print lstm_out.size(),output.size(),final_h.size()
        return F.log_softmax(self.logistic(final_h)), hidden