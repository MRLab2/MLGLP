import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
import pickle as cp
#import _pickle as cp  # python3 compatability
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/../../pytorch_DGCNN' % cur_dir)
import multiprocessing as mp
import torch

from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils import from_networkx



import math
from scipy.sparse import csr_matrix

class GNNGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())

        if len(g.edges()) != 0:
            x, y = list(zip(*g.edges()))
            self.num_edges = len(x)        
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])
        
        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):  
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert(type(list(edge_features.values())[0]) == np.ndarray) 
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in list(edge_features.items())}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)

def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(list(range(len(row))), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i < j and net[i, j] == 0:
            neg[0].append(i)
            neg[1].append(j)
        else:
            continue
    train_neg  = (neg[0][:train_num], neg[1][:train_num])
    test_neg = (neg[0][train_num:], neg[1][train_num:])
    return train_pos, train_neg, test_pos, test_neg

    
def links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, h=1, max_nodes_per_hop=None, node_information=None):
    # automatically select h from {1, 2}
    if h == 'auto':
        # split train into val_train and val_test
        _, _, val_test_pos, val_test_neg = sample_neg(A, 0.1)
        val_A = A.copy()
        val_A[val_test_pos[0], val_test_pos[1]] = 0
        val_A[val_test_pos[1], val_test_pos[0]] = 0
        val_auc_CN = CN(val_A, val_test_pos, val_test_neg)
        val_auc_AA = AA(val_A, val_test_pos, val_test_neg)
        print('\033[91mValidation AUC of AA is {}, CN is {}\033[0m'.format(val_auc_AA, val_auc_CN))
        if val_auc_AA >= val_auc_CN:
            h = 2
            print('\033[91mChoose h=2\033[0m')
        else:
            h = 1
            print('\033[91mChoose h=1\033[0m')

    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    def helper(A, links, g_label):
        '''
        g_list = []
        for i, j in tqdm(zip(links[0], links[1])):
            g, n_labels, n_features = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information)
            max_n_label['value'] = max(max(n_labels), max_n_label['value'])
            g_list.append(GNNGraph(g, g_label, n_labels, n_features))
        return g_list
        '''
        # the new parallel extraction code
        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.map_async(parallel_worker, [((i, j), A, h, max_nodes_per_hop, node_information) for i, j in zip(links[0], links[1])])
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
        pbar.close()
        g_list = [GNNGraph(g, g_label, n_labels, n_features) for g, n_labels, n_features in results]
        max_n_label['value'] = max(max([max(n_labels) for _, n_labels, _ in results]), max_n_label['value'])
        end = time.time()
        print("Time eplased for subgraph extraction: {}s".format(end-start))
        return g_list
        

    print('Enclosing subgraph extraction begins...')
    train_graphs = helper(A, train_pos, 1) + helper(A, train_neg, 0)
    test_graphs = helper(A, test_pos, 1) + helper(A, test_neg, 0)
    print(max_n_label)
    return train_graphs, test_graphs, max_n_label['value']

def parallel_worker(x):
    return subgraph_extraction_labeling(*x)
    
def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None, node_information=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes) 
    subgraph = A[nodes, :][:, nodes]
    # apply node-labeling
    labels = node_label(subgraph)
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph)
    # remove link between target nodes
    if not g.has_edge(0, 1):
        g.add_edge(0, 1)
    return g, labels.tolist(), features


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res

def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    return labels

def AA(A, test_pos, test_neg):
    # Adamic-Adar score
    A_ = A / np.log(A.sum(axis=1))
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    return CalcAUC(sim, test_pos, test_neg)
    
        
def CN(A, test_pos, test_neg):
    # Common Neighbor score
    sim = A.dot(A)
    return CalcAUC(sim, test_pos, test_neg)


def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def single_line(batch_graphs):
    pbar = tqdm(batch_graphs, unit='iteration')
    graphs = []
    for graph in pbar:
        #line_graph, labels = to_line(graph, graph.node_tags)
        line_test(graph, graph.node_tags)
        #graphs.append(line_graph)
    return graphs

def gnn_to_line(batch_graph, max_n_label):
    start = time.time()
    pool = mp.Pool(16)
    #pool = mp.Pool(mp.cpu_count())
    results = pool.map_async(parallel_line_worker, [(graph, max_n_label) for graph in batch_graph])
    remaining = results._number_left
    pbar = tqdm(total=remaining)
    while True:
        pbar.update(remaining - results._number_left)
        if results.ready(): break
        remaining = results._number_left
        time.sleep(1)
    results = results.get()
    pool.close()
    pbar.close()
    g_list = [g for g in results]
    return g_list

def parallel_line_worker(x):
    return to_line(*x)

def to_line(graph, max_n_label):
    edges = graph.edge_pairs
    edge_feas = edge_fea(graph, max_n_label)/2
    edges, feas = to_undirect(edges, edge_feas)
    edges = torch.tensor(edges)
    data = Data(edge_index=edges, edge_attr=feas)
    data.num_nodes = graph.num_nodes
    data = LineGraph()(data)
    data.num_nodes = graph.num_edges
    data['y'] = torch.tensor([graph.label])
    return data

def to_edgepairs(graph):
    x, y = zip(*graph.edges())
    num_edges = len(x)
    edge_pairs = np.ndarray(shape=(num_edges, 2), dtype=np.int32)
    edge_pairs[:, 0] = x
    edge_pairs[:, 1] = y
    edge_pairs = edge_pairs.flatten()
    return edge_pairs

def to_linegraphs(batch_graphs, max_n_label):
    graphs = []
    pbar = tqdm(batch_graphs, unit='iteration')
    for graph in pbar:
        edges = graph.edge_pairs
        edge_feas = edge_fea(graph, max_n_label)/2
        edges, feas = to_undirect(edges, edge_feas)
        edges = torch.tensor(edges)
        data = Data(edge_index=edges, edge_attr=feas)
        data.num_nodes = graph.num_nodes
        data = LineGraph()(data)
        data['y'] = torch.tensor([graph.label])
        data.num_nodes = graph.num_edges
        graphs.append(data)
    return graphs

def edge_fea(graph, max_n_label):
    node_tag = torch.zeros(graph.num_nodes, max_n_label+1)
    tags = graph.node_tags
    tags = torch.LongTensor(tags).view(-1,1)
    node_tag.scatter_(1, tags, 1)
    return node_tag

def edge_fea2(labels, edges):
    feas = []
    for i in range(edges.shape[1]):
        fea = [labels[edges[0][i]], labels[edges[1][i]]]
        fea.sort()
        feas.append(fea)
    feas = np.reshape(feas, [-1, 2])
    feas = np.array([feas[:,0], feas[:,1]], dtype=np.float32)
    return torch.tensor(feas/2)
    
def to_undirect2(edges):
    edges = np.reshape(edges, (-1,2 ))
    sr = np.array([edges[:,0], edges[:,1]], dtype=np.int64)
    rs = np.array([edges[:,1], edges[:,0]], dtype=np.int64)
    target_edge = np.array([[0,1],[1,0]])
    return np.concatenate([target_edge, sr, rs], axis=1)
    
def to_undirect(edges, edge_fea):
    edges = np.reshape(edges, (-1,2 ))
    sr = np.array([edges[:,0], edges[:,1]], dtype=np.int64)
    fea_s = edge_fea[sr[0,:], :]
    fea_s = fea_s.repeat(2,1)
    fea_r = edge_fea[sr[1,:], :]
    fea_r = fea_r.repeat(2,1)
    fea_body = torch.cat([fea_s, fea_r], 1)
    rs = np.array([edges[:,1], edges[:,0]], dtype=np.int64)
    return np.concatenate([sr, rs], axis=1), fea_body


def line_test(graph, label):
    edges = graph.edge_pairs
    edges= to_undirect2(edges)
    feas = edge_fea2(label, edges)
    data = Data(edge_index=torch.tensor(edges), edge_attr=feas.T)
    data = LineGraph()(data)
    elist = data['edge_index'].numpy()
    #elist = [(elist[0][i], elist[1][i]) for i in range(len(elist[0]))]
    #nx_graph = nx.Graph()
    #nx_graph.add_edges_from(elist)
    #return nx_graph, data['x'].numpy()
    #return nx
    
    
###################
class MultiScaleGNNGraph(object):
    def __init__(self, g, multiscales_g, multiscales_g_lable):
        '''
            orginalGNNgraph: orginal GNNgraph
            multiscales: List of graph in different scales
        '''
        self.orginalGNNgraph =g
        self.multiscalegraphs = multiscales_g
        self.lable_multiscalegraphs = multiscales_g_lable

def calculate_labale_base_distance(graph , target_node1, target_node2):
    distances1 = calculate_distances(graph, target_node1)
    distances2 = calculate_distances(graph, target_node2)
   # label = {node: 1+ min(distances1.get(node, 0),distances2.get(node, 0)) +distances1.get(node, 0) + distances2.get(node, 0) for node in set(distances1) | set(distances2)}
    label = {node: 1 + min(distances1.get(node, 0), distances2.get(node, 0)) +
              distances1.get(node, 0) + distances2.get(node, 0)
                if distances1.get(node, 0)!= float('inf')  and distances2.get(node, 0) != float('inf') 
                else 0 for node in set(distances1) | set(distances2)}
    label[target_node1] = 1
    label[target_node2] = 1

    return label

def merge_neighbors(graph, target_node1 , target_node2):
    # neighbors = list(graph.neighbors(target_node))
    neighbors = list(graph.nodes)

    if len(neighbors) == 0:
#         print(f"No neighbors for the target node ")
        return graph
#     distances1 = calculate_distances(graph, target_node1)
#     distances2 = calculate_distances(graph, target_node2)
#    # label = {node: 1+ min(distances1.get(node, 0),distances2.get(node, 0)) +distances1.get(node, 0) + distances2.get(node, 0) for node in set(distances1) | set(distances2)}
#     label = {node: 1 + min(distances1.get(node, 0), distances2.get(node, 0)) +
#               distances1.get(node, 0) + distances2.get(node, 0)
#                 if distances1.get(node, 0)!= float('inf')  and distances2.get(node, 0) != float('inf') 
#                 else 0 for node in set(distances1) | set(distances2)}

#     label[target_node1] = 1
#     label[target_node2] = 1
    label = calculate_labale_base_distance(graph, target_node1 , target_node2)
    # print(distances1)
    # print(distances2)
#     print("lable: ",label)

    #print("started")
    same_distance_neighbors = []
    for neighbor in neighbors:
        if neighbor!= target_node1 and neighbor!= target_node2:
#             print("neighbor:",neighbor)
            for neighbor_of_neighbor in graph.neighbors(neighbor):
#                 print("neighbor_of_neighbor:",neighbor_of_neighbor)
                if neighbor_of_neighbor!= target_node1 and neighbor_of_neighbor!= target_node2 and neighbor_of_neighbor!=neighbor:
#                     print("not in target node")
    #            for node, distance in distances.items():
#                     print("distance_overall:",label[neighbor_of_neighbor], label[neighbor])
                    if label[neighbor_of_neighbor] == label[neighbor]:
                        
#                         print("same distance", neighbor_of_neighbor , neighbor)
                    #if graph.has_edge(target_node, neighbor_of_neighbor):
                        for neighbor_of_neighbor2 in graph.neighbors(neighbor):
                               # if neighbor_of_neighbor2!= target_node and neighbor_of_neighbor2!= neighbor_of_neighbor:
                            if(neighbor_of_neighbor!=neighbor_of_neighbor2):
                                graph.add_edge(neighbor_of_neighbor2, neighbor_of_neighbor) 
#                                 print ("Connect the nodes ",neighbor_of_neighbor2 , neighbor_of_neighbor)
                        graph.remove_node(neighbor)

                        break
    return graph , label

def calculate_distances(graph, target_node):
    distances = {}
    for node in graph.nodes:
        if node == target_node:
            distances[node] = 0
        else:
            try:
                distance = nx.shortest_path_length(graph, source=node, target=target_node)
                distances[node] = distance
            except nx.NetworkXNoPath:
                distances[node] = float('inf')  # If there is no path to the target

    return distances

def get_subgraphs(G, target_node1, target_node2, numbersubgraphs):
    subgraphs = []
    subgraphs_lable = []
    subgraphs.append(G.copy())
    lable = calculate_labale_base_distance(G, target_node1 , target_node2)
    subgraphs_lable.append(lable)

    # Iteratively merge one-hop neighbors of the target node
    for i in range(numbersubgraphs-1):  # You can adjust the number of iterations as needed
#         print("Number of nodes:",(G.nodes))
#         print("List of edges:",(G.edges))
        G , lable_G = merge_neighbors(G, target_node1 , target_node2)
        #convert to utils.GNNGraph
        # Create a mapping from node labels to integer indices
        node_indices = {node: idx for idx, node in enumerate(G.nodes)}
        # Create an empty adjacency matrix
        num_nodes = len(G.nodes)
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        # Populate the adjacency matrix
        for edge in G.edges:
            node1, node2 = edge
            idx1, idx2 = node_indices[node1], node_indices[node2]
            adj_matrix[idx1, idx2] = 1
            adj_matrix[idx2, idx1] = 1  # If the graph is undirected
        net = csr_matrix(adj_matrix)
        # Get the upper triangular part of the sparse matrix
        net_triu = ssp.triu(net, k=1)
        # Get the list of edges from the upper triangular matrix
        edges = list(zip(*net_triu.nonzero()))
        # Create a graph from the edges
        g_gnn = nx.Graph(edges)
        subgraphs.append(g_gnn)
        subgraphs_lable.append(lable_G)
#         print((G.nodes))
    return subgraphs , subgraphs_lable


def subgraphs2multiscalessubgraphs(train_graphs, test_graphs):
    # extract enclosing subgraphs
    def helper_multiscale(graphs):
        multiscaleGraphs_list = []
        for g in tqdm(graphs):
            # Reshape the array into a two-dimensional array
            pairs_array = g.edge_pairs.reshape(-1, 2)
            # Convert each row to a tuple and create a list of pairs
            edge_list_temp = [tuple(row) for row in pairs_array]
            graph = nx.Graph(edge_list_temp)
            target_node1 = 0
            target_node2 = 1
            numbersubgraphs=3
#             print(len(graph.nodes))
            x , l= get_subgraphs(graph,target_node1,target_node2,numbersubgraphs)
            multiscaleGraphs_list.append(MultiScaleGNNGraph(g, x, l) )
        return multiscaleGraphs_list
        
    print('MultiScalesubgraph extraction begins...')
    train_multiscalegraphs = helper_multiscale(train_graphs)
    test_multiscalegraphs = helper_multiscale(test_graphs) 
    return train_multiscalegraphs, test_multiscalegraphs



def ConvertGraphToAdjMatrixFirstTargetNodes(G , target1 , target2):
    # Create a mapping from node labels to integer indices
    # node_indices = {0:0 , 1:1}
    # node_indices = {node: idx for idx, node in enumerate(G.nodes)}
    # Assuming G.nodes contains all the nodes in your graph
    all_nodes_except_01 = [node for node in G.nodes if node not in [target1, target2]]

    # Initialize node_indices with indices for nodes 0 and 1
    node_indices = {target1: 0, target2: 1}

    # Update node_indices with indices for other nodes
    node_indices.update({node: idx for idx, node in enumerate(all_nodes_except_01, start=2)})
    
    # print(node_indices)
    # print(node_indices[3])
    # x1 = node_indices[target1]
    # x2 = node_indices[target2]
    # node_indices[2]
    # node_indices[3]
    # node_indices[target1]=0
    # node_indices[target2]=1
    # Create an empty adjacency matrix
    num_nodes = len(G.nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # Populate the adjacency matrix
    for edge in G.edges:
        node1, node2 = edge
        idx1, idx2 = node_indices[node1], node_indices[node2]
        adj_matrix[idx1, idx2] = 1
        adj_matrix[idx2, idx1] = 1  # If the graph is undirected
    # rowtarget1 = adj_matrix[node_indices[target1],:]
    # rowtarget2 = adj_matrix[node_indices[target2],:]
    # adj_matrix = np.delete(adj_matrix, node_indices[target1], axis=0)
    # adj_matrix = np.delete(adj_matrix, node_indices[target2], axis=0)
    # adj_matrix = np.insert(adj_matrix, 0, rowtarget1, axis=0)
    # adj_matrix = np.insert(adj_matrix, 0, rowtarget2, axis=0)

    # Convert the adjacency matrix to a SciPy sparse matrix (CSR format)
    net = csr_matrix(adj_matrix)
    return net


def linegraphsToMSlinegraphs(linegraphs0, linegraphs1, linegraphs2):
    data = linegraphs0
    # data['edge_index1'] =linegraphs1.edge_index
    # data['num_nodes1'] =linegraphs1.num_nodes
    # data['x1'] =linegraphs1.x
    # data['edge_index2'] =linegraphs2.edge_index
    # data['num_nodes2'] =linegraphs2.num_nodes
    # data['x2'] =linegraphs2.x
    data['sg1'] =linegraphs1
    data['sg2'] =linegraphs2
    return data
class MultiscaleLineGNNGraphForConcatenate(object):
    def __init__(self, multiscales_g0, multiscales_g1, multiscales_g2):
        '''
            orginalGNNgraph: orginal GNNgraph
            multiscalegraphs_0: graph in different scales
            multiscalegraphs_1: graph in different scales
            multiscalegraphs_2: graph in different scales
        '''
        self.MultiscaleLineGraphslist = [multiscales_g0, multiscales_g1,multiscales_g2]
        # self.MultiscaleLineGraphs_0 = multiscales_g0
        # self.MultiscaleLineGraphs_1 = multiscales_g1
        # self.MultiscaleLineGraphs_2 = multiscales_g2
