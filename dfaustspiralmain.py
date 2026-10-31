# -*- coding: utf-8 -*-
'''
@Time    : 4/27/22 
@Author  : Zhang Haoliang
'''
import pickle
import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.backends.cudnn as cudnn
# from models.spiralNetwork import AE
from datasets import dfaustmeshdata
from utils import utils, writer, train_evaldfaust
from torch.utils.data import DataLoader
import trimesh

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
parser.add_argument('--latent_channels', type=int, default=8)
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

args.work_dir = osp.dirname(osp.realpath("__file__"))
args.data_fp = osp.join(args.work_dir, 'data', args.dataset)
args.out_dir = osp.join(args.work_dir, 'out', args.exp_name)
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')

print(args)

utils.makedirs(args.out_dir)
utils.makedirs(args.checkpoints_dir)

writer = writer.Writer(args)
# device = torch.device('cuda', args.device_idx)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(args.n_threads)

# deterministic
torch.manual_seed(args.seed)
# cudnn.benchmark = False
# cudnn.deterministic = True

# load dataset

# meshdata = dfaustmeshdata(args.data_fp)
# train_loader = DataLoader(meshdata.traindata,
#                           batch_size=args.batch_size,
#                           shuffle=True)
# test_loader = DataLoader(meshdata.testdata, batch_size=args.batch_size)
#


# generate/load transform matrices {"mesh":M_verts_faces,'face': F, 'adj': A, 'down_transform': D, 'up_transform': U}
print("Loading Transform Matrices ..")
# transform_fp="template/dfaust/transform_dfaust_old.pkl"
transform_fp="matrix/body_qslim_transform.pkl"
with open(transform_fp, 'rb') as f:
    tmp = pickle.load(f, encoding='latin1')
F=tmp['face']

M=tmp["mesh"]

spiral_indices_list = [
    utils.preprocess_spiral(M[idx][1], args.seq_length[idx],
                            M[idx][0],
                            args.dilation[idx]).to(device)
    for idx in range(len(tmp['face']) - 1)]

edge_index_list = [utils.to_edge_index(adj).to(device) for adj in tmp['adj']]
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
print('Number of parameters: {}'.format(utils.count_parameters(model)))
print(model)



if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                momentum=0.9)
else:
    raise RuntimeError('Use optimizers of SGD or Adam')
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            args.decay_step,
                                            gamma=args.lr_decay)
utils.count_parameters(model)
train_evaldfaust.run(model, train_loader, test_loader, args.epochs, optimizer,
               scheduler, writer, device)

train_evaldfaust.eval_error(model, test_loader, device, meshdata, args.out_dir)
