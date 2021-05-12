import torch


class Args:
    """
    Configuration
    """

    def __init__(self):
        # Clean tensorboard
        self.clean_tensorboard = False

        # Select the device
        self.device = torch.device('cuda:0')

        # Select the method : GraphRNN | GRAN | GraphOpt
        self.note = 'GraphRNN'

        # GraphRNN model parameters
        if 'small' in self.graph_type:
            self.parameter_shrink = 2
        else:
            self.parameter_shrink = 1
        self.hidden_size_node_level_rnn = int(128 / self.parameter_shrink)  # hidden state size for node-level RNN
        self.embedding_size_node_level_rnn = int(64 / self.parameter_shrink)  # the size for LSTM input
        self.hidden_size_edge_level_rnn = 16  # hidden state size for edge level RNN
        self.embedding_size_edge_level_rnn = 8  # the embedding size for output rnn
        self.max_prev_node = None  # max previous node that looks back for GraphRNN ,if none, then auto calculate
        self.num_layers = 4  # Number of layers for GRU

        # Training parameter
        self.data_size = 200
        self.number_vertices = 25
        self.p_erdos = 0.1
        self.k_watts = 10
        self.p_watts = 0.1
        self.m_barabasi=3
