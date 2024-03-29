from random import shuffle
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from tensorboard_logger import configure, log_value
import time as tm
from baselineModels.DGMG.model import *


def train_DGMG(args, dataset_train, model):
    # check if load existing model
    if args.load:
        fname = args.model_save_path + args.fname + 'model_' + str(args.load_epoch) + '.dat'
        model.load_state_dict(torch.load(fname))

        args.lr = 0.00001
        epoch = args.load_epoch
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 1

    # initialize optimizer
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)

    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch <= args.epochs:
        time_start = tm.time()
        # train
        train_DGMG_epoch(epoch, args, model, dataset_train, optimizer, scheduler, is_fast=args.is_fast)
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # print('time used',time_all[epoch - 1])
        # test
        if epoch % args.epochs_test == 0 and epoch >= args.epochs_test_start:
            graphs = test_DGMG_epoch(args,model, is_fast=args.is_fast)
            fname = args.graph_save_path + args.fname_pred + str(epoch) + '.dat'
            save_graph_list(graphs, fname)
            # print('test done, graphs saved')

        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'model_' + str(epoch) + '.dat'
                torch.save(model.state_dict(), fname)
        epoch += 1
    np.save(args.timing_save_path + args.fname, time_all)

def test_DGMG_epoch(args, model, is_fast=False):
    model.eval()
    graph_num = args.test_graph_num

    graphs_generated = []
    for i in range(graph_num):
        # NOTE: when starting loop, we assume a node has already been generated
        node_neighbor = [[]]  # list of lists (first node is zero)
        node_embedding = [Variable(torch.ones(1,args.node_embedding_size)).cuda()] # list of torch tensors, each size: 1*hidden

        node_count = 1
        while node_count<=args.max_num_node:
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
            if a_addnode.data[0][0]==1:
                # print('add node')
                # add node
                node_neighbor.append([])
                node_embedding.append(init_embedding)
                if is_fast:
                    node_embedding_cat = torch.cat(node_embedding, dim=0)
            else:
                break

            edge_count = 0
            while edge_count<args.max_num_node:
                if not is_fast:
                    node_embedding = message_passing(node_neighbor, node_embedding, model)
                    node_embedding_cat = torch.cat(node_embedding, dim=0)
                    graph_embedding = calc_graph_embedding(node_embedding_cat, model)

                # 4 f_addedge
                p_addedge = model.f_ae(graph_embedding)
                a_addedge = sample_tensor(p_addedge)
                # print(a_addedge.data[0][0])

                if a_addedge.data[0][0]==1:
                    # print('add edge')
                    # 5 f_nodes
                    # excluding the last node (which is the new node)
                    node_new_embedding_cat = node_embedding_cat[-1,:].expand(node_embedding_cat.size(0)-1,node_embedding_cat.size(1))
                    s_node = model.f_s(torch.cat((node_embedding_cat[0:-1,:],node_new_embedding_cat),dim=1))
                    p_node = F.softmax(s_node.permute(1,0))
                    a_node = gumbel_softmax(p_node, temperature=0.01)
                    _, a_node_id = a_node.topk(1)
                    a_node_id = int(a_node_id.data[0][0])
                    # add edge
                    node_neighbor[-1].append(a_node_id)
                    node_neighbor[a_node_id].append(len(node_neighbor)-1)
                else:
                    break

                edge_count += 1
            node_count += 1
        # save graph
        node_neighbor_dict = dict(zip(list(range(len(node_neighbor))), node_neighbor))
        graph = nx.from_dict_of_lists(node_neighbor_dict)
        graphs_generated.append(graph)

    return graphs_generated

def train_DGMG_epoch(node_embedding_size, model, dataset, optimizer, scheduler, is_fast = False):
    model.train()
    graph_num = len(dataset)
    order = list(range(graph_num))
    shuffle(order)


    loss_addnode = 0
    loss_addedge = 0
    loss_node = 0
    for i in order:
        model.zero_grad()

        graph = dataset[i]
        # do random ordering: relabel nodes
        node_order = list(range(graph.number_of_nodes()))
        shuffle(node_order)
        order_mapping = dict(zip(graph.nodes(), node_order))
        graph = nx.relabel_nodes(graph, order_mapping, copy=True)

        # NOTE: when starting loop, we assume a node has already been generated
        node_count = 1
        node_embedding = [Variable(torch.ones(1,node_embedding_size)).cuda()] # list of torch tensors, each size: 1*hidden

        loss = 0
        while node_count<=graph.number_of_nodes():
            node_neighbor = graph.subgraph(list(range(node_count))).adjacency_list()  # list of lists (first node is zero)
            node_neighbor_new = graph.subgraph(list(range(node_count+1))).adjacency_list()[-1] # list of new node's neighbors

            # 1 message passing
            # do 2 times message passing
            node_embedding = message_passing(node_neighbor, node_embedding, model)

            # 2 graph embedding and new node embedding
            node_embedding_cat = torch.cat(node_embedding, dim=0)
            graph_embedding = calc_graph_embedding(node_embedding_cat, model)
            init_embedding = calc_init_embedding(node_embedding_cat, model)

            # 3 f_addnode
            p_addnode = model.f_an(graph_embedding)
            if node_count < graph.number_of_nodes():
                # add node
                node_neighbor.append([])
                node_embedding.append(init_embedding)
                if is_fast:
                    node_embedding_cat = torch.cat(node_embedding, dim=0)
                # calc loss
                loss_addnode_step = F.binary_cross_entropy(p_addnode,Variable(torch.ones((1,1))).cuda())
                # loss_addnode_step.backward(retain_graph=True)
                loss += loss_addnode_step
                loss_addnode += loss_addnode_step.data
            else:
                # calc loss
                loss_addnode_step = F.binary_cross_entropy(p_addnode, Variable(torch.zeros((1, 1))).cuda())
                # loss_addnode_step.backward(retain_graph=True)
                loss += loss_addnode_step
                loss_addnode += loss_addnode_step.data
                break


            edge_count = 0
            while edge_count<=len(node_neighbor_new):
                if not is_fast:
                    node_embedding = message_passing(node_neighbor, node_embedding, model)
                    node_embedding_cat = torch.cat(node_embedding, dim=0)
                    graph_embedding = calc_graph_embedding(node_embedding_cat, model)

                # 4 f_addedge
                p_addedge = model.f_ae(graph_embedding)

                if edge_count < len(node_neighbor_new):
                    # calc loss
                    loss_addedge_step = F.binary_cross_entropy(p_addedge, Variable(torch.ones((1, 1))).cuda())
                    # loss_addedge_step.backward(retain_graph=True)
                    loss += loss_addedge_step
                    loss_addedge += loss_addedge_step.data

                    # 5 f_nodes
                    # excluding the last node (which is the new node)
                    node_new_embedding_cat = node_embedding_cat[-1,:].expand(node_embedding_cat.size(0)-1,node_embedding_cat.size(1))
                    s_node = model.f_s(torch.cat((node_embedding_cat[0:-1,:],node_new_embedding_cat),dim=1))
                    p_node = F.softmax(s_node.permute(1,0))
                    # get ground truth
                    a_node = torch.zeros((1,p_node.size(1)))
                    # print('node_neighbor_new',node_neighbor_new, edge_count)
                    a_node[0,node_neighbor_new[edge_count]] = 1
                    a_node = Variable(a_node).cuda()
                    # add edge
                    node_neighbor[-1].append(node_neighbor_new[edge_count])
                    node_neighbor[node_neighbor_new[edge_count]].append(len(node_neighbor)-1)
                    # calc loss
                    loss_node_step = F.binary_cross_entropy(p_node,a_node)
                    # loss_node_step.backward(retain_graph=True)
                    loss += loss_node_step
                    loss_node += loss_node_step.data

                else:
                    # calc loss
                    loss_addedge_step = F.binary_cross_entropy(p_addedge, Variable(torch.zeros((1, 1))).cuda())
                    # loss_addedge_step.backward(retain_graph=True)
                    loss += loss_addedge_step
                    loss_addedge += loss_addedge_step.data
                    break

                edge_count += 1
            node_count += 1

        # update deterministic and lstm
        loss.backward()
        optimizer.step()
        scheduler.step()

    loss_all = loss_addnode + loss_addedge + loss_node

    return loss_all