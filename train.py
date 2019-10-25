import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_value_
import numpy as np
from tensorboard_logger import log_value
from utils import save_model, load_model, get_model_attribute
from train_graph_rnn import evaluate_loss as eval_loss_graph_rnn
from train_dfscode_rnn import evaluate_loss as eval_loss_dfscode_rnn
from dgmg.train_dgmg import evaluate_loss as eval_loss_dgmg

def evaluate_loss(args, model, data, feature_map):
    if args.note == 'GraphRNN':
        loss = eval_loss_graph_rnn(args, model, data, feature_map)
    elif args.note == 'DFScodeRNN':
        loss = eval_loss_dfscode_rnn(args, model, data, feature_map)
    elif args.note == 'DeepGMG':
        loss = eval_loss_dgmg(model, data)

    return loss

# Training epoch for GraphRNN and GraphRNN_MinDFS
def train_epoch(epoch, args, model, dataloader_train, optimizer, scheduler, feature_map):
    # Set training mode for modules
    for _, net in model.items():
        net.train()
        
    batch_count = len(dataloader_train)
    total_loss = 0.0
    for batch_id, data in enumerate(dataloader_train):
        for _, net in model.items():
            net.zero_grad()
        
        loss = evaluate_loss(args, model, data, feature_map)

        loss.backward()
        total_loss += loss.data.item()

        # Clipping gradients
        if args.gradient_clipping:
            for _, net in model.items():
                clip_grad_value_(net.parameters(), 1.0)

        # Update params of rnn and mlp
        for _, opt in optimizer.items():
            opt.step()
        
        for _, sched in scheduler.items():
            sched.step()

        if args.log_tensorboard:
            log_value('train_batch_loss ' + args.fname, loss, batch_id + batch_count * epoch)

    return total_loss / batch_count

def test_data(args, model, dataloader, feature_map):
    for _, net in model.items():
        net.eval()

    batch_count = len(dataloader)
    with torch.no_grad():
        total_loss = 0.0
        for _, data in enumerate(dataloader):
            loss = evaluate_loss(args, model, data, feature_map) 
            total_loss += loss.data.item()
    
    return total_loss / batch_count

# Main training function
def train(args, dataloader_train, model, feature_map, dataloader_validate=None):
    # initialize optimizer
    optimizer = {}
    for name, net in model.items():
        optimizer['optimizer_' + name] = optim.Adam(
            list(net.parameters()), lr=args.lr, weight_decay=5e-5)

    scheduler = {}
    for name, net in model.items():
        optimizer['scheduler_' + name] = MultiStepLR(
            optimizer['optimizer_' + name], milestones=args.milestones, gamma=args.gamma)

    if args.load_model:
        load_model(args.load_model_path, args.device, model, optimizer, scheduler)
        print('Model loaded')

        epoch = get_model_attribute('epoch', args.load_model_path, args.device)
    else:
        epoch = 0
    
    while epoch < args.epochs:
        loss = train_epoch(epoch, args, model, dataloader_train, optimizer, scheduler, feature_map)
        epoch += 1

        # logging
        if args.log_tensorboard:
            log_value('train_loss ' + args.fname, loss, epoch)
        else:
            print('Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))

        # save model checkpoint
        if args.save_model and epoch != 0 and epoch % args.epochs_save == 0:
            save_model(epoch, args, model, optimizer, scheduler, feature_map=feature_map)
            print('Model Saved - Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))
        
        if dataloader_validate is not None and epoch % args.epochs_validate == 0:
            loss_validate = test_data(args, model, dataloader_validate, feature_map)
            if args.log_tensorboard:
                log_value('validate_loss ' + args.fname, loss_validate, epoch)
            else:
                print('Epoch: {}/{}, validation loss: {:.6f}'.format(epoch, args.epochs, loss_validate))

    save_model(epoch, args, model, optimizer, scheduler, feature_map=feature_map)
    print('Model Saved - Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))
