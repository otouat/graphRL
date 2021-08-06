import networkx as nx
import numpy as np
import random
def generate(args):
    graphs=[]
    if args.graph_type=='erdos':
        for i in range(args.data_size):
            graphs.append(nx.erdos_renyi_graph(random.randint(args.num_vertices-50,args.num_vertices),args.p_erdos))
    elif args.graph_type=='watts':
        for i in range(args.data_size):
            graphs.append(nx.watts_strogatz_graph(args.num_vertices,args.k_watts,args.p_watts))
    elif args.graph_type=='barabasi':
        for i in range(150):
            graphs.append(nx.barabasi_albert_graph(random.randint(100,200),4))
    elif args.graph_type=='ladder':
        for i in range(100, 201):
            graphs.append(nx.ladder_graph(i))
        args.max_prev_node = 10
    return graphs