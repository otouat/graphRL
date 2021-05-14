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
        self.note = 'graphRL/baselineModels/GraphRNN'
        self.graph_type = 'erdos'
        # GraphRNN model parameters

        self.hidden_size_node_level_rnn = 128  # hidden state size for node-level RNN
        self.embedding_size_node_level_rnn = 64  # the size for LSTM input
        self.hidden_size_edge_level_rnn = 16  # hidden state size for edge level RNN
        self.embedding_size_edge_level_rnn = 8  # the embedding size for output rnn
        self.max_prev_node = None  # max previous node that looks back for GraphRNN ,if none, then auto calculate
        self.num_layers = 4  # Number of layers for GRU
        self.batch_size = 32  # normal: 32, and the rest should be changed accordingly
        self.test_batch_size = 32
        self.test_total_size = 1000
        self.num_layers = 4

        # Training parameter
        self.data_size = 200
        self.number_vertices = 25
        self.p_erdos = 0.1
        self.k_watts = 10
        self.p_watts = 0.1
        self.m_barabasi = 3

        self.num_workers = 0  # num workers to load data, default 4
        self.batch_ratio = 32  # how many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches
        self.epochs = 100  # now one epoch means self.batch_ratio x batch_size
        self.epochs_test_start = 50
        self.epochs_test = 50
        self.epochs_log = 50
        self.epochs_save = 50

        self.lr = 0.003
        self.milestones = [400, 1000]
        self.lr_rate = 0.3

        #directory
        self.dir_input = "./"
        self.model_save_path = self.dir_input + 'model_save/'  # only for nll evaluation
        self.graph_save_path = self.dir_input + 'graphs/'
        self.figure_save_path = self.dir_input + 'figures/'
        self.timing_save_path = self.dir_input + 'timing/'
        self.figure_prediction_save_path = self.dir_input + 'figures_prediction/'
        self.nll_save_path = self.dir_input + 'nll/'

        self.fname = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_node_level_rnn) + '_'
        self.fname_pred = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_node_level_rnn) + '_pred_'
        self.fname_train = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_node_level_rnn) + '_train_'
        self.fname_test = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_node_level_rnn) + '_test_'

