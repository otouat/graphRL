import torch


class Args:
    """
    Configuration
    """

    def __init__(self):
        # Clean tensorboard
        self.clean_tensorboard = False

        # Select the device
        self.device = 0

        # Select the method : GraphRNN | GRAN | GraphOpt
        self.note = 'GraphRNN'
        self.graph_type = 'erdos'
        # GraphRNN model parameters
        self.parameter_shrink=1
        self.hidden_size_rnn = int(128 / self.parameter_shrink)  # hidden size for main RNN
        self.hidden_size_rnn_output = 16  # hidden size for output RNN
        self.embedding_size_rnn = int(64 / self.parameter_shrink)  # the size for LSTM input
        self.embedding_size_rnn_output = 8  # the embedding size for output rnn
        self.embedding_size_output = int(64 / self.parameter_shrink)  # the embedding size for output (VAE/MLP)
        self.max_num_node = None  # max number of nodes in a graph
        self.max_prev_node = None  # max previous node that looks back
        self.batch_size = 32  # normal: 32, and the rest should be changed accordingly
        self.test_batch_size = 32
        self.test_total_size = 1000
        self.num_layers = 4

        # Training parameter
        self.data_size = 500
        self.num_vertices = 60
        self.p_erdos = 0.1
        self.k_watts = 10
        self.p_watts = 0.1
        self.m_barabasi = 3

        self.num_workers = 0  # num workers to load data, default 4
        self.batch_ratio = 32  # how many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches
        self.epochs = 3000  # now one epoch means self.batch_ratio x batch_size
        self.epochs_test_start = 50
        self.epochs_test = 50
        self.epochs_log = 50
        self.epochs_save = 50

        self.lr = 0.005
        self.milestones = [400, 1000]
        self.lr_rate = 0.3

        #directory
        self.dir_input = "./baselineModels/GraphRNN/ModelData/"
        self.model_save_path = self.dir_input + 'model_save/'  # only for nll evaluation
        self.graph_save_path = self.dir_input + 'graphs/'
        self.figure_save_path = self.dir_input + 'figures/'
        self.timing_save_path = self.dir_input + 'timing/'
        self.figure_prediction_save_path = self.dir_input + 'figures_prediction/'
        self.nll_save_path = self.dir_input + 'nll/'

        self.fname = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_rnn) + '_'
        self.fname_pred = self.note+'_'+self.graph_type+'_'+str(self.num_layers)+'_'+ str(self.hidden_size_rnn)+'_pred_'
        self.fname_train = self.note+'_'+self.graph_type+'_'+str(self.num_layers)+'_'+ str(self.hidden_size_rnn)+'_train_'
        self.fname_test = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_rnn) + '_test_'



        self.load = False  # if load model, default lr is very low
        self.load_epoch = 3000
        self.save = True
