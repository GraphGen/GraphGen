import random
import time
import pickle
from torch.utils.data import DataLoader
from tensorboard_logger import configure
from args import Args
from utils import create_graphs, create_dirs
from preprocess import calc_max_prev_node, dfscodes_weights
from data import Graph_Adj_Matrix_from_file, Graph_DFS_code_from_file
from dgmg.data_dgmg import DGMG_Dataset_from_file
from model import create_model
from train import train

if __name__ == '__main__':
    args = Args()
    args = args.update_args()
    
    create_dirs(args)

    random.seed(123)

    graphs = create_graphs(args)

    random.shuffle(graphs)
    graphs_train = graphs[: int(0.8 * len(graphs))]
    graphs_validate = graphs[int(0.8 * len(graphs)): int(0.9 * len(graphs))]

    # show graphs statistics
    print('Model:', args.note)
    print('Device:', args.device)
    print('Graph type:', args.graph_type)
    print('Training set: {}, Validation set: {}'.format(len(graphs_train), len(graphs_validate)))

    # Loading the feature map
    with open(args.current_dataset_path + 'map.dict', 'rb') as f:
        feature_map = pickle.load(f)

    print('Max number of nodes: {}'.format(feature_map['max_nodes']))
    print('Max number of edges: {}'.format(feature_map['max_edges']))
    print('Min number of nodes: {}'.format(feature_map['min_nodes']))
    print('Min number of edges: {}'.format(feature_map['min_edges']))
    print('Max degree of a node: {}'.format(feature_map['max_degree']))
    print('No. of node labels: {}'.format(len(feature_map['node_forward'])))
    print('No. of edge labels: {}'.format(len(feature_map['edge_forward'])))

    # Setting max_prev_node / max_tail_node and max_head_node
    if args.note == 'GraphRNN':
        start = time.time()
        if args.max_prev_node is None:
            args.max_prev_node = calc_max_prev_node(args.current_processed_dataset_path)
        
        end = time.time()

        print('max_prev_node:', args.max_prev_node)
        print('Time taken to calculate max_prev_node = {}s'.format(end - start))

    if args.note == 'DFScodeRNN' and args.weights:
        feature_map = {
            **feature_map,
            **dfscodes_weights(args.min_dfscode_path, graphs_train, feature_map, args.device)
        }

    # Needed for DataLoader
    collate_fn_train, collate_fn_validate = None, None
    
    if args.note == 'GraphRNN':
        random_bfs = True
        
        dataset_train = Graph_Adj_Matrix_from_file(args, graphs_train, feature_map, random_bfs)
        dataset_validate = Graph_Adj_Matrix_from_file(args, graphs_validate, feature_map, random_bfs)
    
    elif args.note == 'DFScodeRNN':
        dataset_train = Graph_DFS_code_from_file(args, graphs_train, feature_map)
        dataset_validate = Graph_DFS_code_from_file(args, graphs_validate, feature_map)

    elif args.note == 'DeepGMG':
        dataset_train = DGMG_Dataset_from_file(args, graphs_train, feature_map)
        dataset_validate = DGMG_Dataset_from_file(args, graphs_validate, feature_map)
        collate_fn_train = dataset_train.collate_batch
        collate_fn_validate = dataset_validate.collate_batch

        
    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, 
        collate_fn=collate_fn_train)
    dataloader_validate = DataLoader(
        dataset_validate, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, 
        collate_fn=collate_fn_validate)
    
    model = create_model(args, feature_map)

    # Configure tensorboard
    if args.log_tensorboard:
        configure(args.tensorboard_path + args.fname + ' ' + args.time, flush_secs=5)

    train(args, dataloader_train, model, feature_map, dataloader_validate)
