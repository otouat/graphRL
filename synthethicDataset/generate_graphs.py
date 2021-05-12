import networkx as nx
import numpy as np

def generate(args):
    graphs=[]
    if args.graph_type=='erdos':
        for i in range(args.data_size):
            graphs.append(nx.erdos_renyi_graph(args.num_vertices,args.p_erdos))
    elif args.graph_type=='watts':
        for i in range(args.data_size):
            graphs.append(nx.watts_strogatz_graph(args.num_vertices,args.k_watts,args.p_watts))
    elif args.graph_type=='barabasi':
        for i in range(args.data_size):
            graphs.append(nx.barabasi_albert_graph(args.num_vertices,args.m_barabasi))
    return graphs