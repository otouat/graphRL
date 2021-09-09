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

from baselineModels.DGMG.model import *
from baselineModels.DGMG.train import *
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
__all__ = ['DgmgRunner', 'compute_edge_ratio', 'get_graph', 'evaluate']

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


class DgmgRunner(object):

    def __init__(self, config):
        self.config = config
        self.seed = config.seed
        self.dataset_conf = config.dataset
        self.model_conf = config.model
        self.train_conf = config.train
        self.test_conf = config.test
        self.use_gpu = config.use_gpu
        self.gpus = config.gpus
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        model = DGM_graphs(h_size=self.model_conf.node_embedding_size).cuda()
        # initialize optimizer
        optimizer = optim.Adam(list(model.parameters()), lr=self.train_conf.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.train_conf.lr_decay_epoch,
                                                   gamma=self.train_conf.lr_decay)

        # reset gradient
        optimizer.zero_grad()

        epoch = 1
        # Training Loop
        row_list = []
        iter_count = 0

        while epoch <= self.train_conf.max_epoch:
            # train
            train_loss = train_DGMG_epoch(self.model_conf.node_embedding_size, model, self.graphs_train, optimizer,
                                          scheduler, is_fast=self.model_conf.is_fast)
            self.writer.add_scalar('train_loss', train_loss)
            logger.info(
                "NLL Loss @ epoch {:04d} = {}".format(epoch + 1, train_loss))
            # test
            if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
                dict_stat_epoch = self.test_training(model, epoch)
                self.writer.add_scalar('mmd_degree_test', dict_stat_epoch["mmd_degree_test"], iter_count)
                self.writer.add_scalar('mmd_clustering_test', dict_stat_epoch["mmd_clustering_test"], iter_count)
                self.writer.add_scalar('mmd_4orbits_test', dict_stat_epoch["mmd_4orbits_test"], iter_count)
                self.writer.add_scalar('mmd_spectral_test', dict_stat_epoch["mmd_spectral_test"], iter_count)
                self.writer.add_scalar('mmd_degree_dev', dict_stat_epoch["mmd_degree_dev"], iter_count)
                self.writer.add_scalar('mmd_clustering_dev', dict_stat_epoch["mmd_clustering_dev"], iter_count)
                self.writer.add_scalar('mmd_4orbits_dev', dict_stat_epoch["mmd_4orbits_dev"], iter_count)
                self.writer.add_scalar('mmd_spectral_dev', dict_stat_epoch["mmd_spectral_dev"], iter_count)
                row_list.append(dict_stat_epoch)
                logger.info("Saving Snapshot @ epoch {:04d}".format(epoch))
                snapshot(model, optimizer, self.config, epoch + 1,
                         scheduler=scheduler, graph_model="dgmg")
            epoch += 1

        stat_across_epochs = pd.DataFrame(row_list)
        stat_across_epochs.to_csv(open(os.path.join(self.config.save_dir, "stat_across_epochs.csv"), 'wb'), index=False,
                                  header=True)
        save_training_runs(time.strftime('%Y-%b-%d-%H-%M-%S'), self.dataset_conf.name, self.num_graphs,
                           self.model_conf.name,
                           self.train_conf.max_epoch, self.config.save_dir)
        # pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
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
            model = DGM_graphs(h_size=self.model_conf.node_embedding_size).cuda()
            model_file = os.path.join(self.config.save_dir, self.test_conf.test_model_name)
            load_model(model, model_file, self.device)

            model.eval()

            graphs_generated = []
            for i in range(self.test_conf.num_test_gen):
                # NOTE: when starting loop, we assume a node has already been generated
                node_neighbor = [[]]  # list of lists (first node is zero)
                node_embedding = [
                    Variable(torch.ones(1,self.model_conf.node_embedding_size)).cuda()]  # list of torch tensors, each size: 1*hidden

                node_count = 1
                while node_count <= self.max_num_nodes:
                    # 1 message passing
                    # do 2 times message passing
                    node_embedding = message_passing(node_neighbor, node_embedding, model)

                    # 2 graph embedding and new node embedding
                    node_embedding_cat = torch.cat(node_embedding, dim=0)
                    graph_embedding = calc_graph_embedding(node_embedding_cat, model)
                    init_embedding = calc_init_embedding(node_embedding_cat, model)

                    # 3 f_addnode
                    p_addnode = model.f_an(graph_embedding)
                    a_addnode = sample_tensor(p_addnode)
                    # print(a_addnode.data[0][0])
                    if a_addnode.data[0][0] == 1:
                        # print('add node')
                        # add node
                        node_neighbor.append([])
                        node_embedding.append(init_embedding)
                        if self.model_conf.is_fast:
                            node_embedding_cat = torch.cat(node_embedding, dim=0)
                    else:
                        break

                    edge_count = 0
                    while edge_count < self.max_num_nodes:
                        if not self.model_conf.is_fast:
                            node_embedding = message_passing(node_neighbor, node_embedding, model)
                            node_embedding_cat = torch.cat(node_embedding, dim=0)
                            graph_embedding = calc_graph_embedding(node_embedding_cat, model)

                        # 4 f_addedge
                        p_addedge = model.f_ae(graph_embedding)
                        a_addedge = sample_tensor(p_addedge)
                        # print(a_addedge.data[0][0])

                        if a_addedge.data[0][0] == 1:
                            # print('add edge')
                            # 5 f_nodes
                            # excluding the last node (which is the new node)
                            node_new_embedding_cat = node_embedding_cat[-1, :].expand(node_embedding_cat.size(0) - 1,
                                                                                      node_embedding_cat.size(1))
                            s_node = model.f_s(torch.cat((node_embedding_cat[0:-1, :], node_new_embedding_cat), dim=1))
                            p_node = F.softmax(s_node.permute(1, 0))
                            a_node = gumbel_softmax(p_node, temperature=0.01)
                            _, a_node_id = a_node.topk(1)
                            a_node_id = int(a_node_id.data[0][0])
                            # add edge
                            node_neighbor[-1].append(a_node_id)
                            node_neighbor[a_node_id].append(len(node_neighbor) - 1)
                        else:
                            break

                        edge_count += 1
                    node_count += 1
                # save graph
                node_neighbor_dict = dict(zip(list(range(len(node_neighbor))), node_neighbor))
                graph = nx.from_dict_of_lists(node_neighbor_dict)
                graphs_generated.append(graph)

            shuffle(graphs_generated)

        ### Visualize Generated Graphs
        if self.is_vis:
            num_col = self.vis_num_row
            num_row = int(np.ceil(self.num_vis / num_col))
            test_epoch = self.test_conf.test_model_name
            test_epoch = test_epoch[test_epoch.rfind('_') + 1:test_epoch.find('.pth')]
            save_name = os.path.join(self.config.save_dir, '{}_gen_graphs_epoch_{}.png'.format(
                self.config.test.test_model_name[:-4], test_epoch))

            # remove isolated nodes for better visulization
            graphs_pred_vis = [copy.deepcopy(gg) for gg in graphs_generated[:self.num_vis]]

            if self.better_vis:
                for gg in graphs_pred_vis:
                    gg.remove_nodes_from(list(nx.isolates(gg)))

            # display the largest connected component for better visualization
            vis_graphs = []
            for gg in graphs_pred_vis:
                CGs = [gg.subgraph(c) for c in nx.connected_components(gg)]
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
                if len(CGs) == 0:
                    continue
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
        results = defaultdict(float)
        if self.config.dataset.name in ['lobster']:
            acc = eval_acc_lobster_graph(graphs_gen)
            logger.info('Validity accuracy of generated graphs = {}'.format(acc))

        num_nodes_gen = [aa.number_of_nodes() for aa in graphs_generated]

        # Compared with Validation Set
        num_nodes_dev = [gg.number_of_nodes() for gg in self.graphs_dev]  # shape B X 1
        mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev = evaluate(self.graphs_dev, graphs_generated,
                                                                                         degree_only=False)
        mmd_num_nodes_dev = compute_mmd([np.bincount(num_nodes_dev)], [np.bincount(num_nodes_gen)], kernel=gaussian_emd)

        # Compared with Test Set
        num_nodes_test = [gg.number_of_nodes() for gg in self.graphs_test]  # shape B X 1
        mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test = evaluate(self.graphs_test,
                                                                                             graphs_generated,
                                                                                             degree_only=False)
        mmd_num_nodes_test = compute_mmd([np.bincount(num_nodes_test)], [np.bincount(num_nodes_gen)],
                                         kernel=gaussian_emd)
        logger.info(
            "Validation MMD scores of #nodes/degree/clustering/4orbits/spectral are = {}/{}/{}/{}/{}".format(
                mmd_num_nodes_dev, mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev))
        logger.info(
            "Test MMD scores of #nodes/degree/clustering/4orbits/spectral are = {}/{}/{}/{}/{}".format(
                mmd_num_nodes_test, mmd_degree_test, mmd_clustering_test, mmd_4orbits_test,
                mmd_spectral_test))

        return {"mmd_degree_test": mmd_degree_test, "mmd_clustering_test": mmd_clustering_test,
                "mmd_4orbits_test": mmd_4orbits_test, "mmd_spectral_test": mmd_spectral_test,
                "mmd_degree_dev": mmd_degree_dev, "mmd_clustering_dev": mmd_clustering_dev,
                "mmd_4orbits_dev": mmd_4orbits_dev, "mmd_spectral_dev": mmd_spectral_dev}

    def test_training(self, model, epoch_num):

        model.eval()

        graphs_generated = []
        for i in range(self.test_conf.num_test_gen):
            # NOTE: when starting loop, we assume a node has already been generated
            node_neighbor = [[]]  # list of lists (first node is zero)
            node_embedding = [
                Variable(torch.ones(1,
                                    self.model_conf.node_embedding_size)).cuda()]  # list of torch tensors, each size: 1*hidden

            node_count = 1
            while node_count <= self.max_num_nodes:
                # 1 message passing
                # do 2 times message passing
                node_embedding = message_passing(node_neighbor, node_embedding, model)

                # 2 graph embedding and new node embedding
                node_embedding_cat = torch.cat(node_embedding, dim=0)
                graph_embedding = calc_graph_embedding(node_embedding_cat, model)
                init_embedding = calc_init_embedding(node_embedding_cat, model)

                # 3 f_addnode
                p_addnode = model.f_an(graph_embedding)
                a_addnode = sample_tensor(p_addnode)
                # print(a_addnode.data[0][0])
                if a_addnode.data[0][0] == 1:
                    # print('add node')
                    # add node
                    node_neighbor.append([])
                    node_embedding.append(init_embedding)
                    if self.model_conf.is_fast:
                        node_embedding_cat = torch.cat(node_embedding, dim=0)
                else:
                    break

                edge_count = 0
                while edge_count < self.max_num_nodes:
                    if not self.model_conf.is_fast:
                        node_embedding = message_passing(node_neighbor, node_embedding, model)
                        node_embedding_cat = torch.cat(node_embedding, dim=0)
                        graph_embedding = calc_graph_embedding(node_embedding_cat, model)

                    # 4 f_addedge
                    p_addedge = model.f_ae(graph_embedding)
                    a_addedge = sample_tensor(p_addedge)
                    # print(a_addedge.data[0][0])

                    if a_addedge.data[0][0] == 1:
                        # print('add edge')
                        # 5 f_nodes
                        # excluding the last node (which is the new node)
                        node_new_embedding_cat = node_embedding_cat[-1, :].expand(node_embedding_cat.size(0) - 1,
                                                                                  node_embedding_cat.size(1))
                        s_node = model.f_s(torch.cat((node_embedding_cat[0:-1, :], node_new_embedding_cat), dim=1))
                        p_node = F.softmax(s_node.permute(1, 0))
                        a_node = gumbel_softmax(p_node, temperature=0.01)
                        _, a_node_id = a_node.topk(1)
                        a_node_id = int(a_node_id.data[0][0])
                        # add edge
                        node_neighbor[-1].append(a_node_id)
                        node_neighbor[a_node_id].append(len(node_neighbor) - 1)
                    else:
                        break

                    edge_count += 1
                node_count += 1
            # save graph
            node_neighbor_dict = dict(zip(list(range(len(node_neighbor))), node_neighbor))
            graph = nx.from_dict_of_lists(node_neighbor_dict)
            graphs_generated.append(graph)

        shuffle(graphs_generated)
        num_nodes_gen = [aa.number_of_nodes() for aa in graphs_generated]

        # Compared with Validation Set
        num_nodes_dev = [gg.number_of_nodes() for gg in self.graphs_dev]  # shape B X 1
        mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev = evaluate(self.graphs_dev,
                                                                                         graphs_generated,
                                                                                         degree_only=False)
        mmd_num_nodes_dev = compute_mmd([np.bincount(num_nodes_dev)], [np.bincount(num_nodes_gen)], kernel=gaussian_emd)

        # Compared with Test Set
        num_nodes_test = [gg.number_of_nodes() for gg in self.graphs_test]  # shape B X 1
        mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test = evaluate(self.graphs_test,
                                                                                             graphs_generated,
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
