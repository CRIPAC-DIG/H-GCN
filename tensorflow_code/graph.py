import numpy as np

class Graph(object):
    ''' Note: adj_list shows each edge twice. So edge_num is really two times of edge number for undirected graph.'''

    def __init__(self, node_num, edge_num):
        self.node_num = node_num  # n
        self.edge_num = edge_num  # m
        self.adj_list = np.zeros(edge_num, dtype=np.int32) - 1  # a big array for all the neighbors.
        self.adj_idx = np.zeros(node_num + 1,
                                dtype=np.int32)  # idx of the beginning neighbors in the adj_list. Pad one additional element at the end with value equal to the edge_num, i.e., self.adj_idx[-1] = edge_num
        self.adj_wgt = np.zeros(edge_num,
                                dtype=np.float32)  # same dimension as adj_list, wgt on the edge. CAN be float numbers.
        self.node_wgt = np.zeros(node_num, dtype=np.int32)
        self.cmap = np.zeros(node_num, dtype=np.int32) - 1  # mapped to coarser graph

        # weighted degree: the sum of the adjacency weight of each vertex, including self-loop.
        self.degree = np.zeros(node_num, dtype=np.float32)
        self.A = None
        self.C = None  # Matching Matrix

        self.coarser = None
        self.finer = None

    def resize_adj(self, edge_num):
        '''Resize the adjacency list/wgts based on the number of edges.'''
        self.adj_list = np.resize(self.adj_list, edge_num)
        self.adj_wgt = np.resize(self.adj_wgt, edge_num)

    def get_neighs(self, idx):
        '''obtain the list of neigbors given a node.'''
        istart = self.adj_idx[idx]
        iend = self.adj_idx[idx + 1]
        return self.adj_list[istart:iend]

    def get_neigh_edge_wgts(self, idx):
        '''obtain the weights of neighbors given a node.'''
        istart = self.adj_idx[idx]
        iend = self.adj_idx[idx + 1]
        return self.adj_wgt[istart:iend]
