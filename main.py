import pickle
import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T

# from models.Chebnetworks import  AE
from datasets import MeshData
from utils import utils, writer, train_eval,DataLoader

parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--exp_name', type=str, default='interpolation_exp')
# Dfaust
parser.add_argument('--dataset', type=str, default='Dfaust')
parser.add_argument('--split', type=str, default='interpolation')
parser.add_argument('--test_exp', type=str, default='bareteeth')
parser.add_argument('--n_threads', type=int, default=4)
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
parser.add_argument('--batch_size', type=int, default=16)
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

# load dataset  "dfausttemplate.obj"
template_fp = osp.join('template', args.dataset+'template.obj')
meshdata = MeshData(args.data_fp,
                    template_fp,
                    split=args.split,
                    test_exp=args.test_exp)

#                                                   3).std(dim=0)
# Load dfaust dataset
# from datasets import dfaustmeshdata
# meshdata = dfaustmeshdata(args.data_fp)


train_loader = DataLoader(meshdata.train_dataset,batch_size=args.batch_size,shuffle=True)
test_loader = DataLoader(meshdata.test_dataset, batch_size=args.batch_size)

# generate/load transform matrices
transform_fp="mapmatrix/body_qslim_transform.pkl"
with open(transform_fp, 'rb') as f:
    tmp = pickle.load(f, encoding='latin1')

edge_index_list = [utils.to_edge_index(adj).to(device) for adj in tmp['adj']]
down_transform_list = [
    utils.to_sparse(down_transform).to(device)
    for down_transform in tmp['down_transform']
]

# UpSpare_pkl = "UpSpare_array.pkl"
# oldup_transform_list=[
#     utils.to_sparse(up_transform).to(device)
#     for up_transform in tmp['up_transform']
# ]
# with open(UpSpare_pkl, 'rb') as f:
#     up_list = pickle.load(f, encoding='latin1')
# up_transform_list = [
#     utils.to_sparse(up_transform).to(device)
#     for up_transform in up_list
# ]
# up_transform_list[0]=oldup_transform_list[0]

# model = AE(args.in_channels,
#            args.out_channels,
#            args.latent_channels,
#            edge_index_list,
#            down_transform_list,
#            up_transform_list,
#            K=args.K).to(device)
# print(model)
#
# if args.optimizer == 'Adam':
#     optimizer = torch.optim.Adam(model.parameters(),
#                                  lr=args.lr,
#                                  weight_decay=args.weight_decay)
# elif args.optimizer == 'SGD':
#     optimizer = torch.optim.SGD(model.parameters(),
#                                 lr=args.lr,
#                                 weight_decay=args.weight_decay,
#                                 momentum=0.9)
# else:
#     raise RuntimeError('Use optimizers of SGD or Adam')
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                             args.decay_step,
#                                             gamma=args.lr_decay)
#
# train_eval.run(model, train_loader, test_loader, args.epochs, optimizer,
#                scheduler, writer, device)
#
# train_eval.eval_error(model, test_loader, device, meshdata, args.out_dir)
