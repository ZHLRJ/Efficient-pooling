# -*- coding: utf-8 -*-
'''
@Time    : 4/27/22 
@Author  : Zhang Haoliang
'''
# -*- coding: utf-8 -*-
'''
@Time    : 4/26/22 
@Author  : Zhang Haoliang
'''
import pickle
import argparse
import torch
from models.spiralNetwork import AE
from datasets import dfaustmeshdata
from utils import utils, writer, train_evaldfaust
parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--exp_name', type=str, default='interpolation_exp')
parser.add_argument('--dataset', type=str, default='Dfaust')
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--device_idx', type=int, default=0)

# network hyperparameters
parser.add_argument('--out_channels',
                    nargs='+',
                    default=[16,16,16,32],
                    type=int)
parser.add_argument('--latent_channels', type=int, default=10)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--seq_length', type=int, default=[9, 9, 9, 9], nargs='+')
parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')

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

args = parser.parse_args(args=[])
# device = torch.device('cuda', args.device_idx)
def load_spiral_dfaustmodel(PATH ,DU_matrix):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args(args=[])
    transform_fp=DU_matrix
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')
    F = tmp['face']

    M = tmp["mesh"]

    spiral_indices_list = [
        utils.preprocess_spiral(M[idx][1], args.seq_length[idx],
                                M[idx][0],
                                args.dilation[idx]).to(device)
        for idx in range(len(tmp['face']) - 1)
    ]
    down_transform_list = [
        utils.to_sparse(down_transform).to(device)
        for down_transform in tmp['down_transform']
    ]
    up_transform_list = [
        utils.to_sparse(up_transform).to(device)
        for up_transform in tmp['up_transform']
    ]

    model = AE(args.in_channels, args.out_channels, args.latent_channels,
               spiral_indices_list, down_transform_list,
               up_transform_list).to(device)

    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    return model

