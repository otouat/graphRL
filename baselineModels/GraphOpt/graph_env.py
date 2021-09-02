import gym
import itertools
import numpy as np
# import gym_molecule
import copy
import networkx as nx
import random
import time
import matplotlib.pyplot as plt
import csv

class GraphEnv(gym.Env):
    """
    Environment for a general graph
    """
    def __init__(self):
        pass
    def init(self, reward_step_total=1, is_normalize=0,dataset='ba'):
        '''
        own init function, since gym does not support passing argument
        '''
        self.is_normalize = bool(is_normalize)
        self.graph = nx.Graph()
        self.reward_step_total = reward_step_total


        self.counter = 0

        ## load expert data
        if dataset == 'caveman':
            self.dataset = []
            for i in range(2, 3):
                for j in range(6, 11):
                    for k in range(20):
                        self.dataset.append(caveman_special(i, j, p_edge=0.8))  # default 0.8
            self.max_node = 25
            self.max_action = 150
        elif dataset == 'grid':
            self.dataset = []
            for i in range(2, 5):
                for j in range(2, 6):
                    self.dataset.append(nx.grid_2d_graph(i, j))
            self.max_node = 25
            self.max_action = 100
        else:
            print('default dataset: barabasi')
            self.dataset = []
            for i in range(4, 21):
                for j in range(3, 4):
                    for k in range(10):
                        self.dataset.append(nx.barabasi_albert_graph(i, j))
            self.max_node = 25
            self.max_action = 150

        self.action_space = gym.spaces.MultiDiscrete([self.max_node, self.max_node, 3, 2])
        self.observation_space = {}
        self.observation_space['adj'] = gym.Space(shape=[1, self.max_node, self.max_node])
        self.observation_space['node'] = gym.Space(shape=[1, self.max_node, 1])

        self.level = 0  # for curriculum learning, level starts with 0, and increase afterwards

        # compatible with molecule env
        self.max_atom = self.max_node
        self.atom_type_num = 1

    def level_up(self):
        self.level += 1

    def normalize_adj(self, adj):
        degrees = np.sum(adj, axis=2)
        # print('degrees',degrees)
        D = np.zeros((adj.shape[0], adj.shape[1], adj.shape[2]))
        for i in range(D.shape[0]):
            D[i, :, :] = np.diag(np.power(degrees[i, :], -0.5))
        adj_normal = D @ adj @ D
        adj_normal[np.isnan(adj_normal)] = 0
        return adj_normal

    # TODO(Bowen): check
    def step(self, action):
        """

        :param action:
        :return:
        """
        ### init
        info = {}  # info we care about
        self.graph_old = copy.deepcopy(self.graph)
        total_nodes = self.graph.number_of_nodes()

        ### take action
        if action[0, 3] == 0:   # not stop
            stop = False
            if action[0, 1] >= total_nodes:
                self.graph.add_node(int(action[0, 1]))
                self._add_edge(action)
            else:
                self._add_edge(action)  # add new edge
        else:   # stop
            stop = True

        ### calculate intermediate rewards
        # todo: add neccessary rules for the task
        if self.graph.number_of_nodes() + self.graph.number_of_edges()-self.graph_old.number_of_nodes() - \
            self.graph_old.number_of_edges() > 0:
            reward_step = self.reward_step_total / self.max_node
            # successfully added node/edge
        else:
            reward_step = -self.reward_step_total / self.max_node # edge
            self.graph = self.graph_old
            # already exists

        ### calculate and use terminal reward
        if self.graph.number_of_nodes() >= self.max_node - 1 or self.counter >= self.max_action or stop:

            # property rewards
            ## todo: add property reward
            reward_terminal = 1 # arbitrary choice

            new = True  # end of episode
            reward = reward_step + reward_terminal

            # print terminal graph information
            info['final_stat'] = reward_terminal
            info['reward'] = reward
            info['stop'] = stop
        ### use stepwise reward
        else:
            new = False
            reward = reward_step

        # get observation
        ob = self.get_observation()

        self.counter += 1
        if new:
            self.counter = 0

        return ob, reward, new, info

    def reset(self):
        """
        to avoid error, assume a node already exists
        :return: ob
        """
        self.graph.clear()
        self.graph.add_node(0)
        self.counter = 0
        ob = self.get_observation()
        return ob

    # TODO(Bowen): is this necessary
    def render(self, mode='human', close=False):
        return

    # TODO(Bowen): check
    def _add_node(self):
        """

        :param node_type_id:
        :return:
        """
        new_node_idx = self.graph.number_of_nodes()
        self.graph.add_node(new_node_idx)

    # TODO(Bowen): check
    def _add_edge(self, action):
        """

        :param action: [first_node, second_node, edge_type_id]
        :return:
        """

        if self.graph.has_edge(int(action[0,0]), int(action[0,1])) or int(action[0,0])==int(action[0,1]):
            return False
        else:
            self.graph.add_edge(int(action[0,0]), int(action[0,1]))
            return True

    def get_final_graph(self):
        return self.graph

    # TODO(Bowen): check [for featured graph]
    # def get_observation(self):
    #     """
    #
    #     :return: ob, where ob['adj'] is E with dim b x n x n and ob['node']
    #     is F with dim 1 x n x m. NB: n = node_num + node_type_num
    #     """
    #     n = self.graph.number_of_nodes()
    #     n_shift = len(self.possible_node_types)  # assume isolated nodes new nodes exist
    #
    #     d_n = len(self.possible_node_types)
    #     F = np.zeros((1, self.max_node, d_n))
    #
    #     for node in self.graph.nodes_iter(data=True):
    #         node_idx = node[0]
    #         node_type = node[1]['type']
    #         float_array = (node_type == self.possible_node_types).astype(float)
    #         assert float_array.sum() != 0
    #         F[0, node_idx, :] = float_array
    #     temp = F[0, n:n + n_shift, :]
    #     F[0, n:n + n_shift, :] = np.eye(n_shift)
    #
    #     d_e = len(self.possible_edge_types)
    #     E = np.zeros((d_e, self.max_node, self.max_node))
    #     for i in range(d_e):
    #         E[i, :n + n_shift, :n + n_shift] = np.eye(n + n_shift)
    #     for e in self.graph.edges_iter(data=True):
    #         begin_idx = e[0]
    #         end_idx = e[1]
    #         edge_type = e[2]['type']
    #         float_array = (edge_type == self.possible_edge_types).astype(float)
    #         assert float_array.sum() != 0
    #         E[:, begin_idx, end_idx] = float_array
    #         E[:, end_idx, begin_idx] = float_array
    #     ob = {}
    #     if self.is_normalize:
    #         E = self.normalize_adj(E)
    #     ob['adj'] = E
    #     ob['node'] = F
    #     return ob


    # for graphs without features
    def get_observation(self,feature='deg'):
        """

        :return: ob, where ob['adj'] is E with dim b x n x n and ob['node']
        is F with dim 1 x n x m. NB: n = node_num + node_type_num
        """
        n = self.graph.number_of_nodes()
        F = np.zeros((1, self.max_node, 1))
        F[0,:n+1,0] = 1

        E = np.zeros((1, self.max_node, self.max_node))
        E[0,:n,:n] = np.asarray(nx.to_numpy_matrix(self.graph))[np.newaxis,:,:]
        E[0,:n+1,:n+1] += np.eye(n+1)

        ob = {}
        if self.is_normalize:
            E = self.normalize_adj(E)
        ob['adj'] = E
        ob['node'] = F
        return ob

    def get_expert(self, batch_size, is_final=False, curriculum=0,
                   level_total=6, level=0):
        ob = {}
        ob['node'] = np.zeros((batch_size, 1, self.max_node, 1))
        ob['adj'] = np.zeros((batch_size, 1, self.max_node, self.max_node))

        ac = np.zeros((batch_size, 4))
        ### select graph
        dataset_len = len(self.dataset)
        for i in range(batch_size):
            ### get a subgraph
            if curriculum == 1:
                ratio_start = level / float(level_total)
                ratio_end = (level + 1) / float(level_total)
                idx = np.random.randint(int(ratio_start * dataset_len),
                                        int(ratio_end * dataset_len))
            else:
                idx = np.random.randint(0, dataset_len)
            graph = self.dataset[idx]
            edges = graph.edges()
            # select the edge num for the subgraph
            if is_final:
                edges_sub_len = len(edges)
            else:
                edges_sub_len = random.randint(1, len(edges))
            edges_sub = random.sample(edges, k=edges_sub_len)
            graph_sub = nx.Graph(edges_sub)
            graph_sub = max(nx.connected_component_subgraphs(graph_sub),
                            key=len)
            if is_final:  # when the subgraph the whole graph, the expert show
                # stop sign
                node1 = random.randint(0, graph.number_of_nodes() - 1)
                while True:
                    node2 = random.randint(0,graph.number_of_nodes())
                    if node2 != node1:
                        break
                edge_type = 0
                ac[i, :] = [node1, node2, edge_type, 1]  # stop
            else:
                ### random pick an edge from the subgraph, then remove it
                edge_sample = random.sample(graph_sub.edges(), k=1)
                graph_sub.remove_edges_from(edge_sample)
                graph_sub = max(nx.connected_component_subgraphs(graph_sub),
                                key=len)
                edge_sample = edge_sample[0]  # get value
                ### get action
                if edge_sample[0] in graph_sub.nodes() and edge_sample[
                    1] in graph_sub.nodes():
                    node1 = graph_sub.nodes().index(edge_sample[0])
                    node2 = graph_sub.nodes().index(edge_sample[1])
                elif edge_sample[0] in graph_sub.nodes():
                    node1 = graph_sub.nodes().index(edge_sample[0])
                    node2 = graph_sub.number_of_nodes()
                elif edge_sample[1] in graph_sub.nodes():
                    node1 = graph_sub.nodes().index(edge_sample[1])
                    node2 = graph_sub.number_of_nodes()
                else:
                    print('Expert policy error!')
                edge_type = 0
                ac[i, :] = [node1, node2, edge_type, 0]  # don't stop
                # print('action',[node1,node2,edge_type,0])
            # print('action',ac)
            # plt.axis("off")
            # nx.draw_networkx(graph_sub)
            # plt.show()
            ### get observation
            n = graph_sub.number_of_nodes()
            F = np.zeros((1, self.max_node, 1))
            F[0, :n + 1, 0] = 1
            if self.is_normalize:
                ob['adj'][i] = self.normalize_adj(F)
            else:
                ob['node'][i]=F
            # print(F)
            E = np.zeros((1, self.max_node, self.max_node))
            E[0, :n, :n] = np.asarray(nx.to_numpy_matrix(graph_sub))[np.newaxis, :, :]
            E[0, :n + 1, :n + 1] += np.eye(n + 1)
            ob['adj'][i]=E
            # print(E)

        return ob, ac