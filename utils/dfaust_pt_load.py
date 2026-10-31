# -*- coding: utf-8 -*-
'''
@Time    : 4/26/22 
@Author  : Zhang Haoliang
'''
import pickle
import argparse
import numpy as np
import os.path as osp
import torch
from models.Chebnetworks import  AE
from utils import utils, writer, train_evaldfaust
from torch.utils.data import DataLoader
parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--dataset', type=str, default='Dfaust')
parser.add_argument('--device_idx', type=int, default=0)
# network hyperparameters
parser.add_argument('--out_channels',
                    nargs='+',
                    default=[16, 16, 16, 32],
                    type=int)
parser.add_argument('--latent_channels', type=int, default=8)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--K', type=int, default=6)

# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0)
# training hyperparameters
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=300)

# others
parser.add_argument('--seed', type=int, default=1)

# device = torch.device('cuda', args.device_idx)
def load_trained_dfaustmodel(PATH ,DU_matrix):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args(args=[])
    transform_fp=DU_matrix
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')
    edge_index_list = [utils.to_edge_index(adj).to(device) for adj in tmp['adj']]
    down_transform_list = [
        utils.to_sparse(down_transform).to(device)
        for down_transform in tmp['down_transform']
    ]
    up_transform_list = [
        utils.to_sparse(up_transform).to(device)
        for up_transform in tmp['up_transform']
    ]
    model = AE(args.in_channels,
               args.out_channels,
               args.latent_channels,
               edge_index_list,
               down_transform_list,
               up_transform_list,
               K=args.K).to(device)

    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    model.eval()
    return model

