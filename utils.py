import os
import shutil
import pickle
import time
import torch
from process_dataset import produce_graphs_from_raw_format, produce_graphs_from_graphrnn_format, produce_random_walk_sampled_graphs
from preprocess import mapping, graphs_to_min_dfscodes, min_dfscodes_to_tensors

# Routine to create synthetic datasets
def create_graphs(args):
    if 'Lung' in args.graph_type:
        base_path = args.dataset_path + 'Lung/'
        input_path = base_path + 'lung.txt'
    elif 'Breast' in args.graph_type:
        base_path = args.dataset_path + 'Breast/'
        input_path = base_path + 'breast.txt'
    elif 'Leukemia' in args.graph_type:
        base_path = args.dataset_path + 'Leukemia/'
        input_path = base_path + 'leukemia.txt'
    elif 'Yeast' in args.graph_type:
        base_path = args.dataset_path + 'Yeast/'
        input_path = base_path + 'yeast.txt'
    elif 'All' in args.graph_type:
        base_path = args.dataset_path + 'All/'
        input_path = base_path + 'all.txt'
    elif 'ENZYMES' in args.graph_type:
        base_path = args.dataset_path + 'ENZYMES/'
        min_num_nodes = 0
        max_num_nodes = 10000
    elif 'citeseer' in args.graph_type:
        base_path = args.dataset_path + 'citeseer/'
        iterations = 150
    elif 'cora' in args.graph_type:
        base_path = args.dataset_path + 'cora/'
        iterations = 150
    
    if 'inactive' in args.graph_type:
        args.current_dataset_path = base_path + 'inactive_graphs/'
        min_dataset_path = base_path + 'inactive_min_graphs/'
        include_file = base_path + 'actives.txt'
        produce_complementary = True
        args.min_dfscode_path = base_path + 'inactive_min_dfscodes/'
        min_dfscode_tensor_path = base_path + 'inactive_min_dfscode_tensors/'

    elif 'active' in args.graph_type:
        args.current_dataset_path = base_path + 'active_graphs/'
        min_dataset_path = base_path + 'active_min_graphs/'
        include_file = base_path + 'actives.txt'
        produce_complementary = False
        args.min_dfscode_path = base_path + 'active_min_dfscodes/'
        min_dfscode_tensor_path = base_path + 'active_min_dfscode_tensors/'

    else:
        args.current_dataset_path = base_path + 'graphs/'
        min_dataset_path = base_path + 'min_graphs/'
        include_file = None
        produce_complementary = True
        args.min_dfscode_path = base_path + 'min_dfscodes/'
        min_dfscode_tensor_path = base_path + 'min_dfscode_tensors/'
    
    if args.note == 'GraphRNN' or args.note == 'DeepGMG':
        args.current_processed_dataset_path = args.current_dataset_path
    elif args.note == 'DFScodeRNN':
        args.current_processed_dataset_path = min_dfscode_tensor_path

    if args.produce_graphs:
        # Empty the directory
        if os.path.isdir(args.current_dataset_path):
            is_del = input('Delete ' + args.current_dataset_path + ' Y/N:')
            if is_del.strip().lower() == 'y':
                shutil.rmtree(args.current_dataset_path)
            else:
                exit()
        
        os.makedirs(args.current_dataset_path)
        
        if any(graph_type in args.graph_type for graph_type in ['Lung', 'Breast', 'Leukemia', 'Yeast', 'All']):
            count = produce_graphs_from_raw_format(
                input_path, args.current_dataset_path, include_file, produce_complementary=produce_complementary)
        
        elif any(graph_type in args.graph_type for graph_type in ['ENZYMES']):
            count = produce_graphs_from_graphrnn_format(
                base_path, args.graph_type, args.current_dataset_path, 
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes)

        elif any(graph_type in args.graph_type for graph_type in ['cora', 'citeseer']):
            num_factor = 5

            count = produce_random_walk_sampled_graphs(
                base_path, args.graph_type, args.current_dataset_path, 
                iterations=iterations, num_factor=num_factor)
        
        print('Graphs produced', count) 
    else:
        count = len([name for name in os.listdir(args.current_dataset_path) if name.endswith(".dat")])
        print('Graphs counted', count)
    
     # Produce feature map
    feature_map = mapping(args.current_dataset_path, args.current_dataset_path + 'map.dict')

    
    if args.note == 'DFScodeRNN' and args.produce_min_dfscodes:
        # Empty the directory
        mkdir(args.min_dfscode_path)

        start = time.time()
        graphs_to_min_dfscodes(args.current_dataset_path, args.min_dfscode_path, args.current_temp_path)

        end = time.time()
        print('Time taken to make dfscodes = {}s'.format(end - start))

    if args.note == 'DFScodeRNN' and args.produce_min_dfscode_tensors:
        # Empty the directory
        mkdir(min_dfscode_tensor_path)
        
        start = time.time()
        min_dfscodes_to_tensors(args.min_dfscode_path, min_dfscode_tensor_path, feature_map)

        end = time.time()
        print('Time taken to make dfscode tensors= {}s'.format(end - start))

    graphs = [i for i in range(count)]
    return graphs

def mkdir(path):
    if os.path.isdir(path):
        is_del = input('Delete ' + path + ' Y/N:')
        if is_del.strip().lower() == 'y':
            shutil.rmtree(path)
        else:
            exit()

    os.makedirs(path)

def load_graphs(graphs_path, graphs_indices=None):
    """
    Returns a list of graphs given graphs directory and graph indices (Optional)
    If graphs_indices are not provided all graphs will be loaded
    """

    graphs = []
    if graphs_indices is None:
        for name in os.listdir(graphs_path):
            if not name.endswith('.dat'):
                continue

            with open(graphs_path + name, 'rb') as f:
                graphs.append(pickle.load(f))
    else:
        for ind in graphs_indices:
            with open(graphs_path + 'graph' + str(ind) + '.dat', 'rb') as f:
                graphs.append(pickle.load(f))
        
    return graphs

def save_graphs(graphs_path, graphs):
    """
    Save networkx graphs to a directory with indexing starting from 0
    """
    for i in range(len(graphs)):
        with open(graphs_path + 'graph' + str(i) + '.dat', 'wb') as f:
            pickle.dump(graphs[i], f)

# Create Directories for outputs
def create_dirs(args):
    # if args.clean_tensorboard and os.path.isdir(args.tensorboard_path):
    #     shutil.rmtree(args.tensorboard_path)
    
    if args.clean_temp and os.path.isdir(args.temp_path):
        shutil.rmtree(args.temp_path)

    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)

    if not os.path.isdir(args.temp_path):
        os.makedirs(args.temp_path)
    
    if not os.path.isdir(args.tensorboard_path):
        os.makedirs(args.tensorboard_path)
    
    if not os.path.isdir(args.current_temp_path):
        os.makedirs(args.current_temp_path)

def save_model(epoch, args, model, optimizer=None, scheduler=None, **extra_args):
    if not os.path.isdir(args.current_model_save_path):
        os.makedirs(args.current_model_save_path)
    
    fname = args.current_model_save_path + args.fname + '_' + str(epoch) + '.dat'
    checkpoint = {'saved_args': args, 'epoch': epoch}

    save_items = {'model': model}
    if optimizer:
        save_items['optimizer'] = optimizer
    if scheduler:
        save_items['scheduler'] = scheduler

    for name, d in save_items.items():
        save_dict = {}
        for key, value in d.items():
            save_dict[key] = value.state_dict()
        
        checkpoint[name] = save_dict

    if extra_args:
        for arg_name, arg in extra_args.items():
            checkpoint[arg_name] = arg
    
    torch.save(checkpoint, fname)

def load_model(path, device, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location=device)

    for name, d in {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}.items():
        if d is not None:
            for key, value in d.items():
                value.load_state_dict(checkpoint[name][key])
        
        if name == 'model':
            for _, value in d.items():
                value.to(device=device)

def get_model_attribute(attribute, path, device):
    fname = path
    checkpoint = torch.load(fname, map_location=device)

    return checkpoint[attribute]
