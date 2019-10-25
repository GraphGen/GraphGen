import pickle
import os
from multiprocessing import Pool
from functools import partial
import networkx as nx
import numpy as np
import torch
from tqdm.auto import tqdm
from dfs_wrapper import get_min_dfscode

MAX_WORKERS = 48

def mapping(path, dest):
    """
    :param path: path to folder which contains pickled networkx graphs
    :param dest: place where final dictionary pickle file is stored
    :return: dictionary of 4 dictionary which contains forward 
    and backwards mappings of vertices and labels, max_nodes and max_edges
    """

    node_forward, node_backward = {}, {}
    edge_forward, edge_backward = {}, {}
    node_count, edge_count = 0, 0
    max_nodes, max_edges, max_degree = 0, 0, 0
    min_nodes, min_edges = float('inf'), float('inf')

    for filename in tqdm(os.listdir(path)):
        if filename.endswith(".dat"):
            f = open(path + filename, 'rb')
            G = pickle.load(f)
            f.close()
            
            max_nodes = max(max_nodes, len(G.nodes()))
            min_nodes = min(min_nodes, len(G.nodes()))
            for _, data in G.nodes.data():
                if data['label'] not in node_forward:
                    node_forward[data['label']] = node_count
                    node_backward[node_count] = data['label']
                    node_count += 1	

            max_edges = max(max_edges, len(G.edges()))
            min_edges = min(min_edges, len(G.edges()))
            for _, _, data in G.edges.data():
                if data['label'] not in edge_forward:
                    edge_forward[data['label']] = edge_count
                    edge_backward[edge_count] = data['label']
                    edge_count += 1
            
            max_degree = max(max_degree, max([d for n, d in G.degree()]))
                    

    feature_map = {
        'node_forward': node_forward,
        'node_backward': node_backward,
        'edge_forward': edge_forward,
        'edge_backward': edge_backward,
        'max_nodes': max_nodes,
        'min_nodes': min_nodes,
        'max_edges': max_edges,
        'min_edges': min_edges,
        'max_degree': max_degree
    }

    f = open(dest, 'wb')
    pickle.dump(feature_map, f)
    f.close()

    print('Successfully done node count', node_count)					
    print('Successfully done edge count', edge_count)
    
    return feature_map

def get_attributes_len_for_graph_rnn(len_node_map, len_edge_map, max_prev_node):
    """
    Returns (len_node_vec, len_edge_vec, feature_len)
    len_node_vec : Length of vector to represent a node attribute
    len_edge_vec : Length of vector to represent an edge attribute
    num_nodes_to_consider: Number of previous nodes to consider for edges for a given node
    """

    # Last two bits for START node and END node token
    len_node_vec = len_node_map + 2
    # Last three bits in order are NO edge, START egde, END edge token
    len_edge_vec = len_edge_map + 3

    num_nodes_to_consider = max_prev_node
    
    return len_node_vec, len_edge_vec, num_nodes_to_consider

def get_bfs_seq(G, start_id):
    """
    Get a bfs node sequence
    :param G: graph
    :param start_id: starting node
    :return: List of bfs node sequence
    """
    successors_dict = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        succ = []
        for current in start:
            if current in successors_dict:
                succ = succ + successors_dict[current]
        
        output = output + succ
        start = succ
    return output

def get_random_bfs_seq(graph):
    n = len(graph.nodes())
    # Create a random permutaion of graph nodes
    perm = np.random.permutation(n)
    adj = nx.to_numpy_matrix(graph, nodelist=perm, dtype=int)
    G = nx.from_numpy_matrix(adj)

    # Construct bfs ordering starting from a random node
    start_id = np.random.randint(n)
    bfs_seq = get_bfs_seq(G, start_id)

    return [perm[bfs_seq[i]] for i in range(n)]

def graph_to_matrix(graph, node_map, edge_map, max_prev_node=None, random_bfs=False):
    """
    Method for converting graph to a 2d feature matrix
    :param graph: Networkx graph object
    :param node_map: Node label to integer mapping
    :param edge_map: Edge label to integer mapping
    :param max_prev_node: Number of previous nodes to consider for edge prediction
    :random_bfs: Whether or not to do random_bfs
    """
    n = len(graph.nodes())
    len_node_vec, _, num_nodes_to_consider = get_attributes_len_for_graph_rnn(
        len(node_map), len(edge_map), max_prev_node)

    if random_bfs:
        bfs_seq = get_random_bfs_seq(graph)
        bfs_order_map = {bfs_seq[i] : i for i in range(n)}
        graph = nx.relabel_nodes(graph, bfs_order_map)

    # 3D adjacecny matrix in case of edge_features (each A[i, j] is a len_edge_vec size vector)
    adj_mat_2d = torch.ones((n, num_nodes_to_consider))
    adj_mat_2d.tril_(diagonal=-1)
    adj_mat_3d = torch.zeros((n, num_nodes_to_consider, len(edge_map)))

    node_mat = torch.zeros((n, len_node_vec))

    for v, data in graph.nodes.data():
        ind = node_map[data['label']]
        node_mat[v, ind] = 1

    for u, v, data in graph.edges.data():
        if abs(u - v) <= max_prev_node:
            adj_mat_3d[max(u, v), max(u, v) - min(u, v) - 1, edge_map[data['label']]] = 1
            adj_mat_2d[max(u, v), max(u, v) - min(u, v) - 1] = 0
 
    adj_mat = torch.cat(
        (adj_mat_3d, adj_mat_2d.reshape(adj_mat_2d.size(0), adj_mat_2d.size(1), 1), 
        torch.zeros((n, num_nodes_to_consider, 2))), dim=2)
    
    adj_mat = adj_mat.reshape((adj_mat.size(0), -1))

    return torch.cat((node_mat, adj_mat), dim=1)

def graph_to_min_dfscode(graph_file, graphs_path, min_dfscodes_path, temp_path):
    with open(graphs_path + graph_file, 'rb') as f:
        G = pickle.load(f)
        min_dfscode = get_min_dfscode(G, temp_path)

        if len(G.edges()) == len(min_dfscode):
            with open(min_dfscodes_path + graph_file, 'wb') as f:
                pickle.dump(min_dfscode, f)
        else:
            print('Error in min dfscode for filename', graph_file)
            exit()

def graphs_to_min_dfscodes(graphs_path, min_dfscodes_path, temp_path):
    """
    :param graphs_path: Path to directory of graphs in networkx format
    :param min_dfscodes_path: Path to directory to store the min dfscodes
    :param temp_path: path for temporary files
    :return: length of dataset
    """
    graphs = []
    for filename in os.listdir(graphs_path):
        if filename.endswith(".dat"):
            graphs.append(filename)
    
    with Pool(processes=MAX_WORKERS) as pool:
        for _ in tqdm(pool.imap_unordered(
            partial(graph_to_min_dfscode, graphs_path=graphs_path, min_dfscodes_path=min_dfscodes_path,
            temp_path=temp_path), graphs, chunksize=16)):
            pass
        

    print('Done creating min dfscodes')

def dfscode_to_tensor(dfscode, feature_map):
    max_nodes, max_edges = feature_map['max_nodes'], feature_map['max_edges']
    node_forward_dict, edge_forward_dict = feature_map['node_forward'], feature_map['edge_forward']
    num_nodes_feat, num_edges_feat = len(feature_map['node_forward']), len(feature_map['edge_forward'])

    # max_nodes, num_nodes_feat and num_edges_feat are end token labels
    # So ignore tokens are one higher
    dfscode_tensors = {
        't1': (max_nodes + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        't2': (max_nodes + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'v1': (num_nodes_feat + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'e': (num_edges_feat + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'v2': (num_nodes_feat + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'len': len(dfscode)
    }

    for i, code in enumerate(dfscode):
        dfscode_tensors['t1'][i] = int(code[0])
        dfscode_tensors['t2'][i] = int(code[1])
        dfscode_tensors['v1'][i] = int(node_forward_dict[code[2]])
        dfscode_tensors['e'][i] = int(edge_forward_dict[code[3]])
        dfscode_tensors['v2'][i] = int(node_forward_dict[code[4]])
    
    # Add end token
    dfscode_tensors['t1'][len(dfscode)], dfscode_tensors['t2'][len(dfscode)] = max_nodes, max_nodes
    dfscode_tensors['v1'][len(dfscode)], dfscode_tensors['v2'][len(dfscode)] = num_nodes_feat, num_nodes_feat
    dfscode_tensors['e'][len(dfscode)] = num_edges_feat

    return dfscode_tensors

def dfscode_from_file_to_tensor_to_file(min_dfscode_file, min_dfscodes_path, min_dfscode_tensors_path, feature_map):
    with open(min_dfscodes_path + min_dfscode_file, 'rb') as f:
        min_dfscode = pickle.load(f)
    
    dfscode_tensors = dfscode_to_tensor(min_dfscode, feature_map)  
    
    with open(min_dfscode_tensors_path + min_dfscode_file, 'wb') as f:
        pickle.dump(dfscode_tensors, f)

def min_dfscodes_to_tensors(min_dfscodes_path, min_dfscode_tensors_path, feature_map):
    """
    :param min_dfscodes_path: Path to directory of pickled min dfscodes
    :param min_dfscode_tensors_path: Path to directory to store the min dfscode tensors
    :param feature_map:
    :return: length of dataset
    """
    min_dfscodes = []
    for filename in os.listdir(min_dfscodes_path):
        if filename.endswith(".dat"):
            min_dfscodes.append(filename)
    
    with Pool(processes=MAX_WORKERS) as pool:
        for _ in tqdm(pool.imap_unordered(
            partial(dfscode_from_file_to_tensor_to_file, min_dfscodes_path=min_dfscodes_path,
            min_dfscode_tensors_path=min_dfscode_tensors_path, feature_map=feature_map), 
            min_dfscodes, chunksize=16)):
            pass

def calc_max_prev_node_helper(idx, graphs_path):
    with open(graphs_path + 'graph' + str(idx) + '.dat', 'rb') as f:
        G = pickle.load(f)

    max_prev_node = []
    for _ in range(100):
        bfs_seq = get_random_bfs_seq(G)
        bfs_order_map = {bfs_seq[i] : i for i in range(len(G.nodes()))}
        G = nx.relabel_nodes(G, bfs_order_map)
        
        max_prev_node_iter = 0
        for u, v in G.edges():
            max_prev_node_iter = max(max_prev_node_iter, max(u, v) - min(u, v))

        max_prev_node.append(max_prev_node_iter)

    return max_prev_node
    
def calc_max_prev_node(graphs_path):
    """
    Approximate max_prev_node from simulating bfs sequences 
    """
    max_prev_node = []
    count = len([name for name in os.listdir(graphs_path) if name.endswith(".dat")])

    max_prev_node = []
    with Pool(processes=MAX_WORKERS) as pool:
        for max_prev_node_g in tqdm(pool.imap_unordered(
            partial(calc_max_prev_node_helper, graphs_path=graphs_path), list(range(count)))):
            max_prev_node.extend(max_prev_node_g)
    
    max_prev_node = sorted(max_prev_node)[-1 * int(0.001 * len(max_prev_node))]
    return max_prev_node

def dfscodes_weights(dataset_path, graph_list, feature_map, device):
    freq = {
        't1_freq': torch.ones(feature_map['max_nodes'] + 1, device=device),
        't2_freq': torch.ones(feature_map['max_nodes'] + 1, device=device),
        'v1_freq': torch.ones(len(feature_map['node_forward']) + 1, device=device),
        'e_freq': torch.ones(len(feature_map['edge_forward']) + 1, device=device),
        'v2_freq': torch.ones(len(feature_map['node_forward']) + 1, device=device)
    }

    for idx in graph_list:
        with open(dataset_path + 'graph' + str(idx) + '.dat', 'rb') as f:
            min_dfscode = pickle.load(f)
            for code in min_dfscode:
                freq['t1_freq'][int(code[0])] += 1
                freq['t2_freq'][int(code[1])] += 1
                freq['v1_freq'][feature_map['node_forward'][code[2]]] += 1
                freq['e_freq'][feature_map['edge_forward'][code[3]]] += 1
                freq['v2_freq'][feature_map['node_forward'][code[4]]] += 1
            
    freq['t1_freq'][-1] = len(graph_list)
    freq['t2_freq'][-1] = len(graph_list)
    freq['v1_freq'][-1] = len(graph_list)
    freq['e_freq'][-1] = len(graph_list)
    freq['v2_freq'][-1] = len(graph_list)

    print('Weights computed')

    return {
        't1_weight': torch.pow(torch.torch.max(freq['t1_freq']), 0.3) / torch.pow(freq['t1_freq'], 0.3),
        't2_weight': torch.pow(torch.max(freq['t2_freq']), 0.3) / torch.pow(freq['t2_freq'], 0.3),
        'v1_weight': torch.pow(torch.max(freq['v1_freq']), 0.3) / torch.pow(freq['v1_freq'], 0.3),
        'e_weight': torch.pow(torch.max(freq['e_freq']), 0.3) / torch.pow(freq['e_freq'], 0.3),
        'v2_weight': torch.pow(torch.max(freq['v2_freq']), 0.3) / torch.pow(freq['v2_freq'], 0.3)
    }

def random_walk_with_restart_sampling(
    G, start_node, iterations, fly_back_prob=0.15
):

    sampled_graph = nx.Graph()
    sampled_graph.add_node(start_node, label=G.nodes[start_node]['label'])

    curr_node = start_node

    for _ in range(iterations):
        edges = [n for n in G.neighbors(curr_node)]
        index_of_edge = np.random.randint(0, len(edges))
        chosen_node = edges[index_of_edge]

        sampled_graph.add_node(chosen_node, label=G.nodes[chosen_node]['label'])
        sampled_graph.add_edge(curr_node, chosen_node, label=G.edges[curr_node, chosen_node]['label'])

        choice = np.random.choice(['prev', 'neigh'], 1, p=[fly_back_prob, 1 - fly_back_prob])
        if choice == 'neigh':
            curr_node = chosen_node
        else:
            curr_node = start_node

    # sampled_graph = G.subgraph(sampled_node_set)
    
    return sampled_graph
