import pickle
import torch
from torch.utils.data import Dataset
from preprocess import graph_to_matrix, dfscode_to_tensor, get_attributes_len_for_graph_rnn
from dfs_wrapper import get_min_dfscode

class Graph_Adj_Matrix_from_file(Dataset):
    """
    Dataset for reading graphs from files and returning adjacency like matrices
    :param args: Args object
    :param graph_list: List of graph indices to be included in the dataset
    :param feature_map: feature_map for the dataset generated by the mapping
    :random_bfs: Whether or not to do random_bfs
    """
    def __init__(self, args, graph_list, feature_map, random_bfs=False):
        # Path to folder containing dataset
        self.dataset_path = args.current_processed_dataset_path
        self.graph_list = graph_list
        self.feature_map = feature_map
        self.max_prev_node = args.max_prev_node # No. of previous nodes to consider for edge prediction
        self.random_bfs = random_bfs
        
        self.max_nodes = feature_map['max_nodes']
        
        len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len_for_graph_rnn(
            len(feature_map['node_forward']), len(feature_map['edge_forward']), self.max_prev_node)
        
        self.feature_len = len_node_vec + num_nodes_to_consider * len_edge_vec

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        with open(self.dataset_path + 'graph' + str(self.graph_list[idx]) + '.dat', 'rb') as f:
            G = pickle.load(f)
        
        x_item = torch.zeros((self.max_nodes, self.feature_len))

        # get feature matrix for the graph
        adj_feature_mat = graph_to_matrix(
            G, self.feature_map['node_forward'], self.feature_map['edge_forward'], 
            self.max_prev_node, self.random_bfs)
        
        # prepare x_item
        x_item[0:adj_feature_mat.shape[0], :adj_feature_mat.shape[1]] = adj_feature_mat
        
        return {'x': x_item, 'len': len(adj_feature_mat)}

class Graph_Adj_Matrix(Dataset):
    """
    Mainly for testing purposes
    Dataset for taking graphs list and returning adjacency like matrices
    :param graph_list: List of graphs to be included in the dataset
    :param feature_map: feature_map for the dataset generated by the mapping
    :param max_prev_node: No of previous nodes to consider for edge prediction
    :random_bfs: Whether or not to do random_bfs
    """
    def __init__(self, graph_list, feature_map, max_prev_node, random_bfs=False):
        self.graph_list = graph_list
        self.feature_map = feature_map
        self.max_prev_node = max_prev_node
        self.random_bfs = random_bfs

        self.max_nodes = feature_map['max_nodes']
        
        len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len_for_graph_rnn(
            len(feature_map['node_forward']), len(feature_map['edge_forward']), self.max_prev_node)
        
        self.feature_len = len_node_vec + num_nodes_to_consider * len_edge_vec

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        G = self.graph_list[idx]

        x_item = torch.zeros((self.max_nodes, self.feature_len))

        # get feature matrix for the graph
        adj_feature_mat = graph_to_matrix(
            G, self.feature_map['node_forward'], self.feature_map['edge_forward'], 
            self.max_prev_node, self.random_bfs)
        
        # prepare x_item
        x_item[0:adj_feature_mat.shape[0], :adj_feature_mat.shape[1]] = adj_feature_mat
        
        return {'x': x_item, 'len': len(adj_feature_mat)}

class Graph_DFS_code_from_file(Dataset):
    """
    Dataset for reading graphs from files and returning matrices
    corresponding to dfs code entries
    :param args: Args object
    :param graph_list: List of graph indices to be included in the dataset
    :param feature_map: feature_map for the dataset generated by the mapping
    """
    def __init__(self, args, graph_list, feature_map):
        # Path to folder containing dataset
        self.dataset_path = args.current_processed_dataset_path
        self.graph_list = graph_list
        self.feature_map = feature_map
        self.temp_path = args.current_temp_path

        self.max_edges = feature_map['max_edges']
        max_nodes, len_node_vec, len_edge_vec = feature_map['max_nodes'], len(feature_map['node_forward']) + 1, len(feature_map['edge_forward']) + 1
        self.feature_len = 2 * (max_nodes + 1) + 2 * len_node_vec + len_edge_vec
        
    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        with open(self.dataset_path + 'graph' + str(self.graph_list[idx]) + '.dat', 'rb') as f:
            dfscode_tensors = pickle.load(f)

        return dfscode_tensors

class Graph_DFS_code(Dataset):
    """
    Mainly for testing purposes
    Dataset for returning matrices corresponding to dfs code entries
    :param args: Args object
    :param graph_list: List of graph indices to be included in the dataset
    :param feature_map: feature_map for the dataset generated by the mapping
    """
    def __init__(self, args, graph_list, feature_map):
        # Path to folder containing dataset
        self.graph_list = graph_list
        self.feature_map = feature_map
        self.temp_path = args.current_temp_path

        self.max_edges = feature_map['max_edges']
        max_nodes, len_node_vec, len_edge_vec = feature_map['max_nodes'], len(feature_map['node_forward']) + 1, len(feature_map['edge_forward']) + 1
        self.feature_len = 2 * (max_nodes + 1) + 2 * len_node_vec + len_edge_vec
        
    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        G = self.graph_list[idx]

        # Get DFS code matrix
        min_dfscode = get_min_dfscode(G, self.temp_path)
        print(min_dfscode)

        # dfscode tensors
        dfscode_tensors = dfscode_to_tensor(min_dfscode, self.feature_map)
        
        return dfscode_tensors