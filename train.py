import sys
import argparse
import math
import copy
import os
from os.path import join as pjoin

from dataloader import get_cifar10, get_cifar100
from utils import accuracy, alpha_weight, get_distance_loss, get_similarity, plot, interleave, de_interleave, get_pair_indices, bha_coeff, bha_coeff_loss

from model.wrn import WideResNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import accuracy

from torch import Tensor
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train (model, datasets, dataloaders, modelpath,
          criterion, optimizer, scheduler, validation, test, args):

    if not os.path.isdir(modelpath):
        os.makedirs(modelpath)
    model_subpath = 'cifar10' if args.num_classes == 10 else 'cifar100'

    training_loss = 1e8
    validation_loss = 1e8
    test_loss = 1e8

    if validation:
        best_model = {
            'epoch': 0,
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
            'training_losses': [],
            'validation_losses': [],
            'test_losses': [],
            'model_depth' : args.model_depth,
            'num_classes' : args.num_classes,
            'num_labeled' : args.num_labeled,
            'num_validation' : args.num_validation,
            'model_width' : args.model_width,
            'drop_rate' : args.drop_rate,
        } 
    
    # access datasets and dataloders
    labeled_dataset = datasets['labeled']
    labeled_loader = dataloaders['labeled']
    unlabeled_loader = dataloaders['unlabeled']
    unlabeled_dataset = datasets['unlabeled']
    if validation:
        validation_dataset = datasets['validation']
        validation_loader = dataloaders['validation']
    if test:
        test_dataset = datasets['test']
        test_loader = dataloaders['test']

    print('Training started with clipping')
    print('Dropping last not full batch')
    print('-' * 20)
    model.train()
    # train
    training_losses = []
    validation_losses = []
    test_losses = []
    for epoch in range(args.epoch):
        running_loss = 0.0
        model.train()
        for i in range(args.iter_per_epoch):
            try:
                (x_l, x_l_s), y_l = next(labeled_loader)
                
            except StopIteration:
                labeled_loader = iter(DataLoader(labeled_dataset,
                                                 batch_size=args.train_batch,
                                                 shuffle=True,
                                                 num_workers=args.num_workers,
                                                 drop_last=True))
                (x_l, x_l_s), y_l = next(labeled_loader)
            y_l = y_l.to(device)
            try:
                (x_ul_w, x_ul_s), _ = next(unlabeled_loader)
            except StopIteration:
                unlabeled_loader = iter(DataLoader(unlabeled_dataset,
                                                batch_size=args.train_batch*args.mu,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                drop_last=True))
                (x_ul_w, x_ul_s), _ = next(unlabeled_loader)
    
            # mix all batches to do a single forward pass
            # 1 batch labeled, 1 batch strong aug labeled
            # args.mu batches weak aug unlabeled, args.mu batches strong aug unlabeled
            # print("x_l: ", x_l.size())
            # print("x_l_s: ", x_l_s.size())
            # print("x_ul_w: ", x_ul_w.size())
            # print("x_ul_s: ", x_ul_s.size())
            inputs = interleave(torch.cat((x_l, x_l_s, x_ul_w, x_ul_s)), 2*args.mu+2).to(device)
            outputs = model(inputs)

            # split batches after computing logits
            outputs = de_interleave(outputs, 2*args.mu+2)
            output_l = outputs[:args.train_batch]
            output_l_s = outputs[args.train_batch:args.train_batch*2]
            output_ul_w, output_ul_s = outputs[args.train_batch*2:].chunk(2)
            del outputs

            # calculate loss for labeled data
            l_loss = criterion(output_l, y_l)

            # calculate supervised pair loss
            pair_loss_s = compute_pair_loss(output_l_s, y_l, args.confidence_threshold, args.similarity_threshold, mode='supervised')

            # get the pseudo-label from weak augmented unlabeled data 
            target_ul = F.softmax(output_ul_w, dim=1)
            hot_target_ul = torch.where(target_ul > args.confidence_threshold, 1, 0)
            idx, y_pl = torch.where(hot_target_ul == 1)
            # get the corresping strong labeled images to compute the loss
            output_pl = output_ul_s[idx]
            output_pl = output_pl.to(device)

            # calculate loss for pseudo-labeled data 
            pl_loss = 0.0 if (output_pl.size(0) == 0) else criterion(output_pl, y_pl)
          

            # calculate unsupervised pair loss
            pair_loss_u = compute_pair_loss(output_ul_s, target_ul, args.confidence_threshold, args.similarity_threshold, mode='unsupervised')
            

            total_loss = (l_loss +  args.lambda_u*pl_loss + args.lambda_pair_s*pair_loss_s + args.lambda_pair_u*pair_loss_u)

            # back propagation
            optimizer.zero_grad()
            total_loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

            #Gradient Value Clipping
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()

            running_loss += total_loss.item()
        '''
        print('l_loss ', l_loss)
        print('pl_loss ', pl_loss)
        print('pair_loss_s ', pair_loss_s)
        print('pair_loss_u ', pair_loss_u)
        '''
        training_loss = running_loss/(args.iter_per_epoch)
        training_losses.append(training_loss)
        print('Epoch: {} : Train Loss : {:.5f} '.format(
            epoch, training_loss))
        
        # Calculate loss for validation set every epoch
        # Save the best model
        running_loss = 0.0
        if validation:
            model.eval()
            for (x_val, _), y_val in validation_loader:
                with torch.no_grad():
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    output_val = model(x_val)
                    loss = criterion(output_val, y_val)

                    running_loss += loss.item() * x_val.size(0)

            validation_loss = running_loss / len(validation_dataset)
            validation_losses.append(validation_loss)
            print('Epoch: {} : Validation Loss : {:.5f} '.format(
            epoch, validation_loss))

            if best_model['epoch'] == 0 or validation_loss < best_model['validation_losses'][-1]:
                best_model = {
                    'epoch': epoch,
                    'model_state_dict': copy.deepcopy(model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                    'training_losses':  copy.deepcopy(training_losses),
                    'validation_losses': copy.deepcopy(validation_losses),
                    'test_losses': copy.deepcopy(test_losses),
                    'model_depth' : args.model_depth,
                    'num_classes' : args.num_classes,
                    'num_labeled' : args.num_labeled,
                    'num_validation' : args.num_validation,
                    'model_width' : args.model_width,
                    'drop_rate' : args.drop_rate
                }
                torch.save(best_model, pjoin(modelpath, 'best_model_{}_{}.pt'.format(model_subpath, args.num_labeled)))
                print('Best model updated with validation loss : {:.5f} '.format(validation_loss))
        # update learning rate
        scheduler.step()
        print("new lr: ", scheduler.get_lr())
        # Check test error with current model over test dataset
        running_loss = 0.0
        if test:
            total_accuracy = []
            test_loss = 0.0
            model.eval()
            for x_test, y_test in test_loader:
                with torch.no_grad():
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    output_test = model(x_test)                              
                    loss = criterion(output_test, y_test)
                    running_loss += loss.item() * x_test.size(0)
                    acc = accuracy(output_test, y_test)
                    total_accuracy.append(sum(acc))
            test_loss = running_loss / len(test_dataset)
            test_losses.append(test_loss)
            print('Epoch: {} : Test Loss : {:.5f} '.format(
                epoch, test_loss))
            print('Accuracy of the network on test images: %d %%' % (
                sum(total_accuracy)/len(total_accuracy)))

    last_model = {
        'epoch': epoch,
        'model_state_dict': copy.deepcopy(model.state_dict()),
        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
        'training_losses':  copy.deepcopy(training_losses),
        'validation_losses': copy.deepcopy(validation_losses),
        'test_losses': copy.deepcopy(test_losses),
        'model_depth' : args.model_depth,
        'num_classes' : args.num_classes,
        'num_labeled' : args.num_labeled,
        'num_validation' : args.num_validation,
        'model_width' : args.model_width,
        'drop_rate' : args.drop_rate
    }
    torch.save(last_model, pjoin(modelpath, 'last_model_{}_{}.pt'.format(model_subpath, args.num_labeled)))
    if validation:
        # recover better weights from validation
        model.load_state_dict(best_model['model_state_dict'])
    return model

def compute_pair_loss(logits: Tensor, targets: Tensor, confidence_threshold = 0.95, similarity_threshold = 0.9, mode = 'unsupervised'):
    if(mode == 'supervised'):
        targets_s = np.zeros(logits.size())
        targets_s = np.full_like(targets_s, 1e-10)
        targets_idx = [np.arange(targets_s.shape[0]), targets.tolist()]
        targets_s[targets_idx] = 1 - (1e-10 * (targets_s.shape[1]-1))
        targets = torch.tensor(targets_s).to(device)
    indices = get_pair_indices(targets, ordered_pair=True)
    total_size = len(indices) // 2

    i_indices, j_indices = indices[:, 0], indices[:, 1]
    targets_max_prob = targets.max(dim=1).values
    targets_i_max_prob = targets_max_prob[i_indices]

    logits_j = logits[j_indices]
    targets_i = targets[i_indices]
    targets_j = targets[j_indices]

    # conf_mask should not track gradient
    conf_mask = (targets_i_max_prob > confidence_threshold).detach().float()

    similarities: Tensor = get_similarity(targets_i=targets_i,
                                                targets_j=targets_j)
    # sim_mask should not track gradient
    sim_mask = F.threshold(similarities, similarity_threshold, 0).detach()

    distance = get_distance_loss(logits=logits_j,
                                        targets=targets_i)

    loss = conf_mask * sim_mask * distance

    loss = torch.sum(loss) / total_size

    return loss