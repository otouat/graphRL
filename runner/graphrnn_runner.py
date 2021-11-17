"""Copyright (c) 2019 Conan"""

from __future__ import (division, print_function)
import os
import time
from random import shuffle

import networkx as nx
import numpy as np
import copy
import pickle
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.utils.data.distributed as distributed

from baselineModels.GraphRNN.model import *
from baselineModels.GraphRNN.data import *
from baselineModels.GraphRNN.train import *
from utils.logger import get_logger
from utils.train_helper import data_to_gpu, snapshot, load_model, EarlyStopper
from utils.data_helper import *
from utils.eval_helper import *
from utils.dist_helper import compute_mmd, gaussian_emd, gaussian, emd, gaussian_tv
from utils.vis_helper import draw_graph_list, draw_graph_list_separate
from utils.data_parallel import DataParallel

try:
    ###
    # workaround for solving the issue of multi-worker
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (10000, rlimit[1]))
    ###
except:
    pass

logger = get_logger('exp_logger')
__all__ = ['GraphRnnRunner', 'compute_edge_ratio', 'get_graph', 'evaluate']

NPR = np.random.RandomState(seed=1234)
save_file = 'save_model_learning.csv'


def compute_edge_ratio(G_list):
    num_edges_max, num_edges = .0, .0
    for gg in G_list:
        num_nodes = gg.number_of_nodes()
        num_edges += gg.number_of_edges()
        num_edges_max += num_nodes ** 2

    ratio = (num_edges_max - num_edges) / num_edges
    return ratio


def get_graph(adj):
    """ get a graph from zero-padded adj """
    # remove all zeros rows and columns
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    return G


def evaluate(graph_gt, graph_pred, degree_only=True):
    mmd_degree = degree_stats(graph_gt, graph_pred)

    if degree_only:
        mmd_4orbits = 0.0
        mmd_clustering = 0.0
        mmd_spectral = 0.0
    else:
        mmd_4orbits = orbit_stats_all(graph_gt, graph_pred)
        mmd_clustering = clustering_stats(graph_gt, graph_pred)
        mmd_spectral = spectral_stats(graph_gt, graph_pred)

    return mmd_degree, mmd_clustering, mmd_4orbits, mmd_spectral


def save_training_runs(date, dataset, dataset_num, model, epochs, file_dir):
    data = [{"Date": date, "dataset_name": dataset, "dataset_num_graphs": dataset_num, "model_name": model,
             "num_epochs": epochs, "file_dir": file_dir}]

    if not os.path.isfile(save_file):
        df = pd.DataFrame(data)
        df.to_csv(save_file)
        return 1

    df = pd.DataFrame(data)
    df.to_csv(save_file, mode='a', header=False)

    return 1


class GraphRnnRunner(object):

    def __init__(self, config):
        self.config = config
        self.seed = config.seed
        self.dataset_conf = config.dataset
        self.model_conf = config.model
        self.train_conf = config.train
        self.test_conf = config.test
        self.use_gpu = config.use_gpu
        self.gpus = config.gpus
        self.device = config.device
        self.writer = SummaryWriter(config.save_dir)
        self.is_vis = config.test.is_vis
        self.better_vis = config.test.better_vis
        self.num_vis = config.test.num_vis
        self.vis_num_row = config.test.vis_num_row
        self.is_single_plot = config.test.is_single_plot
        self.num_gpus = len(self.gpus)
        self.is_shuffle = True

        assert self.use_gpu == True

        if self.train_conf.is_resume:
            self.config.save_dir = self.train_conf.resume_dir

        ### load graphs
        self.graphs = create_graphs(config.dataset.name, data_dir=config.dataset.data_path)

        self.train_ratio = config.dataset.train_ratio
        self.dev_ratio = config.dataset.dev_ratio
        # self.block_size = config.model.block_size
        # self.stride = config.model.sample_stride
        self.num_graphs = len(self.graphs)
        self.num_train = int(float(self.num_graphs) * self.train_ratio)
        self.num_dev = int(float(self.num_graphs) * self.dev_ratio)
        self.num_test_gt = self.num_graphs - self.num_train
        self.num_test_gen = config.test.num_test_gen

        logger.info('Train/val/test = {}/{}/{}'.format(self.num_train, self.num_dev,
                                                       self.num_test_gt))

        ### shuffle all graphs
        if self.is_shuffle:
            self.npr = np.random.RandomState(self.seed)
            self.npr.shuffle(self.graphs)

        self.graphs_train = self.graphs[:self.num_train]
        self.graphs_dev = self.graphs[:self.num_dev]
        self.graphs_test = self.graphs[self.num_train:]

        self.config.dataset.sparse_ratio = compute_edge_ratio(self.graphs_train)
        logger.info('No Edges vs. Edges in training set = {}'.format(
            self.config.dataset.sparse_ratio))

        self.num_nodes_pmf_train = np.bincount([gg.number_of_nodes() for gg in self.graphs_train])
        self.max_num_nodes = len(self.num_nodes_pmf_train)
        self.num_nodes_pmf_train = self.num_nodes_pmf_train / self.num_nodes_pmf_train.sum()

        ### save split for benchmarking
        if config.dataset.is_save_split:
            base_path = os.path.join(config.dataset.data_path, 'save_split')
            if not os.path.exists(base_path):
                os.makedirs(base_path)

            save_graph_list(
                self.graphs_train,
                os.path.join(base_path, '{}_train.p'.format(config.dataset.name)))
            save_graph_list(
                self.graphs_dev,
                os.path.join(base_path, '{}_dev.p'.format(config.dataset.name)))
            save_graph_list(
                self.graphs_test,
                os.path.join(base_path, '{}_test.p'.format(config.dataset.name)))

    def train(self):
        ### create data loader
        # train_dataset = eval(self.dataset_conf.loader_name)(self.config, self.graphs_train, tag='train')
        max_prev_node = self.model_conf.max_prev_node
        if self.dataset_conf.node_order == "BFS":
            dataset = Graph_to_sequence(self.graphs_train, max_prev_node=self.model_conf.max_prev_node,
                                        max_num_node=self.max_num_nodes)
            max_prev_node = dataset.max_prev_node
        elif self.dataset_conf.node_order == "BFSMAX":
            max_prev_node = self.max_num_nodes - 1
            dataset = Graph_to_sequence(self.graphs_train, max_prev_node=max_prev_node,
                                        max_num_node=self.max_num_nodes)
        elif self.dataset_conf.node_order == "DFS":
            max_prev_node = self.max_num_nodes - 1
            dataset = Graph_to_sequence_dfs(self.graphs_train, max_prev_node,
                                            max_num_node=self.max_num_nodes)
        elif self.dataset_conf.node_order == "nobfs":
            max_prev_node = self.max_num_nodes - 1
            dataset = Graph_to_sequence_nobfs(self.graphs_train, max_num_node=self.max_num_nodes)
        elif self.dataset_conf.node_order == "degree_descent":
            max_prev_node = self.max_num_nodes - 1
            dataset = Graph_to_sequence_dfs(self.graphs_train, max_prev_node,order=self.dataset_conf.node_order,
                                            max_num_node=self.max_num_nodes)
        elif self.dataset_conf.node_order == "RCM":
            dataset = Graph_to_sequence_rcm(self.graphs_train, max_prev_node=self.model_conf.max_prev_node,
                                        max_num_node=self.max_num_nodes)
            max_prev_node = dataset.max_prev_node
        sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
            [1.0 / len(dataset) for i in range(len(dataset))],
            num_samples=self.model_conf.batch_size * self.model_conf.batch_ratio,
            replacement=True)
        self.model_conf.max_prev_node = max_prev_node
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.model_conf.batch_size,
            num_workers=self.train_conf.num_workers,
            sampler=sample_strategy
        )

        if self.model_conf.is_mlp:
            rnn = RNN(input_size=int(max_prev_node),
                      embedding_size=int(self.model_conf.embedding_size_rnn),
                      hidden_size=int(self.model_conf.hidden_size_rnn), num_layers=int(self.model_conf.num_layers),
                      has_input=True,
                      has_output=False).cuda()

            output = MLP_plain(h_size=int(self.model_conf.hidden_size_rnn),
                               embedding_size=int(self.model_conf.embedding_size_output),
                               y_size=int(max_prev_node)).cuda()
        else:
            rnn = RNN(input_size=int(max_prev_node),
                      embedding_size=int(self.model_conf.embedding_size_rnn),
                      hidden_size=int(self.model_conf.hidden_size_rnn), num_layers=int(self.model_conf.num_layers),
                      has_input=True,
                      has_output=True, output_size=int(self.model_conf.hidden_size_rnn_output)).cuda()

            output = RNN(input_size=1, embedding_size=int(self.model_conf.embedding_size_rnn_output),
                         hidden_size=int(self.model_conf.hidden_size_rnn_output),
                         num_layers=int(self.model_conf.num_layers),
                         has_input=True,
                         has_output=True, output_size=1).cuda()

        # create optimizer
        params_rnn = filter(lambda p: p.requires_grad, rnn.parameters())
        params_output = filter(lambda p: p.requires_grad, output.parameters())

        optimizer_rnn = optim.Adam(params_rnn, lr=self.train_conf.lr)
        optimizer_output = optim.Adam(params_output, lr=self.train_conf.lr)

        scheduler_rnn = optim.lr_scheduler.MultiStepLR(optimizer_rnn, self.train_conf.lr_decay_epoch,
                                                       gamma=self.train_conf.lr_decay)
        scheduler_output = optim.lr_scheduler.MultiStepLR(optimizer_output, self.train_conf.lr_decay_epoch,
                                                          gamma=self.train_conf.lr_decay)

        resume_epoch = 0
        if self.train_conf.is_resume:
            rnn_file = os.path.join(self.train_conf.resume_dir,
                                    self.train_conf.resume_rnn_name)
            output_file = os.path.join(self.train_conf.resume_dir,
                                       self.train_conf.resume_output_name)
            load_model(
                rnn,
                rnn_file,
                self.device,
                optimizer=optimizer_rnn,
                scheduler=scheduler_rnn)
            load_model(
                output,
                output_file,
                self.device,
                optimizer=optimizer_output,
                scheduler=scheduler_output)
            resume_epoch = self.train_conf.resume_epoch
        # Training Loop
        row_list = []
        iter_count = 0
        for epoch in range(resume_epoch, self.train_conf.max_epoch):

            if self.model_conf.is_mlp:
                train_loss, iter_count = train_mlp_epoch_runner(iter_count, rnn, output, train_loader,
                                                                optimizer_rnn, optimizer_output,
                                                                scheduler_rnn, scheduler_output)
            else:
                train_loss, iter_count = train_rnn_epoch_runner(iter_count, rnn, output, train_loader,
                                                                optimizer_rnn, optimizer_output,
                                                                scheduler_rnn, scheduler_output,
                                                                self.config.model.num_layers)

            self.writer.add_scalar('train_loss', train_loss, iter_count)

            logger.info(
                "NLL Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count, train_loss))

            # snapshot model
            if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
                dict_stat_epoch = self.test_training(rnn, output, epoch + 1)
                self.writer.add_scalar('mmd_degree_test', dict_stat_epoch["mmd_degree_test"], iter_count)
                self.writer.add_scalar('mmd_clustering_test', dict_stat_epoch["mmd_clustering_test"], iter_count)
                self.writer.add_scalar('mmd_4orbits_test', dict_stat_epoch["mmd_4orbits_test"], iter_count)
                self.writer.add_scalar('mmd_spectral_test', dict_stat_epoch["mmd_spectral_test"], iter_count)
                self.writer.add_scalar('mmd_degree_dev', dict_stat_epoch["mmd_degree_dev"], iter_count)
                self.writer.add_scalar('mmd_clustering_dev', dict_stat_epoch["mmd_clustering_dev"], iter_count)
                self.writer.add_scalar('mmd_4orbits_dev', dict_stat_epoch["mmd_4orbits_dev"], iter_count)
                self.writer.add_scalar('mmd_spectral_dev', dict_stat_epoch["mmd_spectral_dev"], iter_count)
                row_list.append(dict_stat_epoch)
                logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
                snapshot(rnn, optimizer_rnn, self.config, epoch + 1,
                         scheduler=scheduler_rnn, graph_model="rnn")
                snapshot(output, optimizer_output, self.config, epoch + 1,
                         scheduler=scheduler_output, graph_model="output")

            torch.cuda.empty_cache()
        stat_across_epochs = pd.DataFrame(row_list)

        stat_across_epochs.to_csv(os.path.join(self.config.save_dir, "stat_across_epochs.csv"), index=False,
                                  header=True)
        save_training_runs(time.strftime('%Y-%b-%d-%H-%M-%S'), self.dataset_conf.name, self.num_graphs,
                           self.model_conf.name,
                           self.train_conf.max_epoch, self.config.save_dir)
        self.writer.close()

        return 1

    def test(self):
        self.config.save_dir = self.test_conf.test_model_dir

        ### Compute Erdos-Renyi baseline
        if self.config.test.is_test_ER:
            p_ER = sum([aa.number_of_edges() for aa in self.graphs_train]) / sum(
                [aa.number_of_nodes() ** 2 for aa in self.graphs_train])
            graphs_gen = [nx.fast_gnp_random_graph(self.max_num_nodes, p_ER, seed=ii) for ii in
                          range(self.num_test_gen)]
        else:
            ### load model
            # create models
            if self.model_conf.is_mlp:
                rnn = RNN(input_size=int(self.model_conf.max_prev_node),
                          embedding_size=int(self.model_conf.embedding_size_rnn),
                          hidden_size=int(self.model_conf.hidden_size_rnn), num_layers=int(self.model_conf.num_layers),
                          has_input=True,
                          has_output=False).cuda()

                output = MLP_plain(h_size=int(self.model_conf.hidden_size_rnn),
                                   embedding_size=int(self.model_conf.embedding_size_output),
                                   y_size=int(self.model_conf.max_prev_node)).cuda()
            else:
                rnn = RNN(input_size=int(self.model_conf.max_prev_node),
                          embedding_size=int(self.model_conf.embedding_size_rnn),
                          hidden_size=int(self.model_conf.hidden_size_rnn), num_layers=int(self.model_conf.num_layers),
                          has_input=True,
                          has_output=True, output_size=int(self.model_conf.hidden_size_rnn_output)).cuda()

                output = RNN(input_size=1, embedding_size=int(self.model_conf.embedding_size_rnn_output),
                             hidden_size=int(self.model_conf.hidden_size_rnn_output),
                             num_layers=int(self.model_conf.num_layers),
                             has_input=True,
                             has_output=True, output_size=1).cuda()

            # create optimizer
            rnn_file = os.path.join(self.config.save_dir, self.test_conf.test_rnn_name)
            output_file = os.path.join(self.config.save_dir, self.test_conf.test_output_name)
            load_model(rnn, rnn_file, self.device)
            load_model(output, output_file, self.device)

            rnn.eval()
            output.eval()
            num_test_batch = int(np.ceil(len(self.graphs) / self.test_conf.batch_size))
            G_pred = []
            for i in tqdm(range(num_test_batch)):
                with torch.no_grad():
                    if self.model_conf.is_mlp:
                        graphs_gen = test_mlp_epoch_runner(self.train_conf.max_epoch, self.model_conf, rnn, output,
                                                           test_batch_size=self.test_conf.batch_size)
                        G_pred.extend(graphs_gen)
                    else:
                        graphs_gen = test_rnn_epoch_runner(self.train_conf.max_epoch, self.model_conf, rnn, output,
                                                           test_batch_size=self.test_conf.batch_size)
                        G_pred.extend(graphs_gen)

            shuffle(G_pred)
        ### Visualize Generated Graphs
        if self.is_vis:
            num_col = self.vis_num_row
            num_row = int(np.ceil(self.num_vis / num_col))
            test_epoch = self.test_conf.test_rnn_name
            test_epoch = test_epoch[test_epoch.rfind('_') + 1:test_epoch.find('.pth')]
            save_name = os.path.join(self.config.save_dir, '{}_gen_graphs_epoch_{}.png'.format(
                self.config.test.test_rnn_name[:-4], test_epoch))

            # remove isolated nodes for better visualization
            graphs_pred_vis = [copy.deepcopy(gg) for gg in G_pred[:self.num_vis]]

            if self.better_vis:
                for gg in graphs_pred_vis:
                    gg.remove_nodes_from(list(nx.isolates(gg)))

            # display the largest connected component for better visualization
            vis_graphs = []
            for gg in graphs_pred_vis:
                CGs = [gg.subgraph(c) for c in nx.connected_components(gg)]
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
                vis_graphs += [CGs[0]]

            if self.is_single_plot:
                draw_graph_list(vis_graphs, num_row, num_col, fname=save_name, layout='spring')
            else:
                draw_graph_list_separate(vis_graphs, fname=save_name[:-4], is_single=True, layout='spring')

            save_name = os.path.join(self.config.save_dir, 'train_graphs.png')

            if self.is_single_plot:
                draw_graph_list(
                    self.graphs_train[:self.num_vis],
                    num_row,
                    num_col,
                    fname=save_name,
                    layout='spring')
            else:
                draw_graph_list_separate(
                    self.graphs_train[:self.num_vis],
                    fname=save_name[:-4],
                    is_single=True,
                    layout='spring')

        ### Evaluation
        if self.config.dataset.name in ['lobster']:
            acc = eval_acc_lobster_graph(graphs_gen)
            logger.info('Validity accuracy of generated graphs = {}'.format(acc))

        num_nodes_gen = [len(aa) for aa in G_pred]

        # Compared with Test Set
        num_nodes_test = [len(gg.nodes) for gg in self.graphs]  # shape B X 1
        mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test = evaluate(self.graphs,
                                                                                             G_pred,
                                                                                             degree_only=False)
        mmd_num_nodes_test = compute_mmd([np.bincount(num_nodes_test)], [np.bincount(num_nodes_gen)],
                                         kernel=gaussian_emd)

        logger.info(
            "Test MMD scores of #nodes/degree/clustering/4orbits/spectral are = {}/{}/{}/{}/{}".format(
                mmd_num_nodes_test, mmd_degree_test, mmd_clustering_test, mmd_4orbits_test,
                mmd_spectral_test))

        return {"mmd_degree_test": mmd_degree_test, "mmd_clustering_test": mmd_clustering_test,
                "mmd_4orbits_test": mmd_4orbits_test, "mmd_spectral_test": mmd_spectral_test}

    def test_training(self, rnn, output, epoch_num):

        rnn.eval()
        output.eval()
        num_test_batch = int(np.ceil(len(self.graphs_dev) / self.test_conf.batch_size))
        G_pred = []
        for i in tqdm(range(num_test_batch)):
            with torch.no_grad():
                if self.model_conf.is_mlp:
                    graphs_gen = test_mlp_epoch_runner(self.train_conf.max_epoch, self.model_conf, rnn, output,
                                                       test_batch_size=self.test_conf.batch_size)
                    G_pred.extend(graphs_gen)
                else:
                    graphs_gen = test_rnn_epoch_runner(self.train_conf.max_epoch, self.model_conf, rnn, output,
                                                       test_batch_size=self.test_conf.batch_size)
                    G_pred.extend(graphs_gen)

        shuffle(G_pred)

        num_nodes_gen = [len(aa) for aa in G_pred]

        # Compared with Validation Set
        num_nodes_dev = [gg.number_of_nodes() for gg in self.graphs_dev]  # shape B X 1
        mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev = evaluate(self.graphs_dev, G_pred,
                                                                                         degree_only=False)
        mmd_num_nodes_dev = compute_mmd([np.bincount(num_nodes_dev)], [np.bincount(num_nodes_gen)], kernel=gaussian_emd)

        # Compared with Test Set
        num_nodes_test = [gg.number_of_nodes() for gg in self.graphs_test]  # shape B X 1
        mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test = evaluate(self.graphs_test,
                                                                                             G_pred,
                                                                                             degree_only=False)
        mmd_num_nodes_test = compute_mmd([np.bincount(num_nodes_test)], [np.bincount(num_nodes_gen)],
                                         kernel=gaussian_emd)
        logger.info(
            "@ epoch {:04d} Validation MMD scores of #nodes/degree/clustering/4orbits/spectral are = {}/{}/{}/{}/{}".format(
                epoch_num, mmd_num_nodes_dev, mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev))
        logger.info(
            "@ epoch {:04d} Test MMD scores of #nodes/degree/clustering/4orbits/spectral are = {}/{}/{}/{}/{}".format(
                epoch_num, mmd_num_nodes_test, mmd_degree_test, mmd_clustering_test, mmd_4orbits_test,
                mmd_spectral_test))

        return {"epoch_num": epoch_num, "mmd_degree_test": mmd_degree_test, "mmd_clustering_test": mmd_clustering_test,
                "mmd_4orbits_test": mmd_4orbits_test, "mmd_spectral_test": mmd_spectral_test,
                "mmd_degree_dev": mmd_degree_dev, "mmd_clustering_dev": mmd_clustering_dev,
                "mmd_4orbits_dev": mmd_4orbits_dev, "mmd_spectral_dev": mmd_spectral_dev}

    def generate_graphs(self):
        self.config.save_dir = self.test_conf.test_model_dir

        ### Compute Erdos-Renyi baseline
        if self.config.test.is_test_ER:
            p_ER = sum([aa.number_of_edges() for aa in self.graphs_train]) / sum(
                [aa.number_of_nodes() ** 2 for aa in self.graphs_train])
            graphs_gen = [nx.fast_gnp_random_graph(self.max_num_nodes, p_ER, seed=ii) for ii in
                          range(self.num_test_gen)]
        else:
            ### load model
            # create models
            if self.model_conf.is_mlp:
                rnn = RNN(input_size=int(self.model_conf.max_prev_node),
                          embedding_size=int(self.model_conf.embedding_size_rnn),
                          hidden_size=int(self.model_conf.hidden_size_rnn), num_layers=int(self.model_conf.num_layers),
                          has_input=True,
                          has_output=False).cuda()

                output = MLP_plain(h_size=int(self.model_conf.hidden_size_rnn),
                                   embedding_size=int(self.model_conf.embedding_size_output),
                                   y_size=int(self.model_conf.max_prev_node)).cuda()
            else:
                rnn = RNN(input_size=int(self.model_conf.max_prev_node),
                          embedding_size=int(self.model_conf.embedding_size_rnn),
                          hidden_size=int(self.model_conf.hidden_size_rnn), num_layers=int(self.model_conf.num_layers),
                          has_input=True,
                          has_output=True, output_size=int(self.model_conf.hidden_size_rnn_output)).cuda()

                output = RNN(input_size=1, embedding_size=int(self.model_conf.embedding_size_rnn_output),
                             hidden_size=int(self.model_conf.hidden_size_rnn_output),
                             num_layers=int(self.model_conf.num_layers),
                             has_input=True,
                             has_output=True, output_size=1).cuda()

            # create optimizer
            rnn_file = os.path.join(self.config.save_dir, self.test_conf.test_rnn_name)
            output_file = os.path.join(self.config.save_dir, self.test_conf.test_output_name)
            load_model(rnn, rnn_file, self.device)
            load_model(output, output_file, self.device)

            rnn.eval()
            output.eval()
            num_test_batch = int(np.ceil(len(self.graphs) / self.test_conf.batch_size))
            G_pred = []
            for i in tqdm(range(num_test_batch)):
                with torch.no_grad():
                    if self.model_conf.is_mlp:
                        graphs_gen = test_mlp_epoch_runner(self.train_conf.max_epoch, self.model_conf, rnn, output,
                                                           test_batch_size=self.test_conf.batch_size)
                        G_pred.extend(graphs_gen)
                    else:
                        graphs_gen = test_rnn_epoch_runner(self.train_conf.max_epoch, self.model_conf, rnn, output,
                                                           test_batch_size=self.test_conf.batch_size)
                        G_pred.extend(graphs_gen)

            shuffle(G_pred)

        base_path = os.path.join(self.config.dataset.data_path, 'generated_graphs')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        if self.model_conf.is_mlp:
            save_graph_list(
                G_pred,
                os.path.join(base_path, '{}_{}_{}.p'.format(self.config.dataset.name,self.model_conf.name+"_MLP",self.dataset_conf.node_order)))
        else :
            save_graph_list(
                G_pred,
                os.path.join(base_path, '{}_{}_{}.p'.format(self.config.dataset.name, self.model_conf.name,
                                                            self.dataset_conf.node_order)))
        return G_pred
