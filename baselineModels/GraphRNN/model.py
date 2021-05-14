import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from collections import OrderedDict
import math
import numpy as np
import time


class RNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, has_input=True, has_output=False,
                 output_size=None):
        super(RNN, self).__init__()
        self.num_layer = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )
        self.relu = nn.ReLU()

        self.hidden = None

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.25)
            elif 'weigth' in name:
                nn.init.xavier_uniform(param, gain=nn.init.calculate_gain('sigmoid'))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layer, batch_size, self.hidden_size)).cuda()

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            input = self.input(input_raw)
            input = self.input(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input=input, lengths=input_len, batch_first=True)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(sequence=output_raw, batch_first=True)[0]
        if self.has_output:
            output_raw = self.output(output_raw)
        return output_raw

def create_model(args):

    node_level_rnn = RNN()
    edge_level_rnn= RNN()

    model ={
        'node_level_rnn':node_level_rnn,
        'edge_level_rnn':edge_level_rnn
    }