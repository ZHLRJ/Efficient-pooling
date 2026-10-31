# -*- coding: utf-8 -*-
'''
@Time    : 10/18/22 
@Author  : Zhang Haoliang
'''
import pickle
import argparse
import os
import os.path as osp
import pickle
import trimesh
import torch
import os
from demoModel.CoMAdataset.PAIConv.utils import get_adj,sparse_mx_to_torch_sparse_tensor
import numpy as np
import scipy.sparse as sp
from demoModel.CoMAdataset.PAIConv.PaiConv import PaiAutoencoder

# from models.Chebnetworks import  AE
from datasets import MeshData
from utils import utils, writer, train_eval,DataLoader
ds_factors = [4, 4, 4] #, 4]
kernal_size = [9, 9, 9, 9] #, 9]
step_sizes = [2, 2, 1, 1, 1]
filter_sizes_enc = [3, 16, 32, 64] #, 128]
filter_sizes_dec = [64, 32, 32, 16, 3]
nz=16
lr=1e-3
regularization= 5e-5

parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--exp_name', type=str, default='interpolation_exp')
# Dfaust
parser.add_argument('--dataset', type=str, default='Dfaust')
parser.add_argument('--split', type=str, default='interpolation')
parser.add_argument('--test_exp', type=str, default='bareteeth')
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--device_idx', type=int, default=0)

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=300)
# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0)

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
transform_fp="demoModel/CoMAdataset/PAIConv/PAIdownsampling_matrices.pkl"
with open(transform_fp, 'rb') as f:
	downsampling_matrices = pickle.load(f, encoding='latin1')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

M_verts_faces = downsampling_matrices['M_verts_faces']
# if meshpackage == 'trimesh':
M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process=False) for i in
     range(len(M_verts_faces))]
A = downsampling_matrices['A']
D = downsampling_matrices['D']
U = downsampling_matrices['U']
F = downsampling_matrices['F']

vertices = [torch.cat([torch.tensor(M_verts_faces[i][0], dtype=torch.float32), torch.zeros((1, 3), dtype=torch.float32)], 0).to(device) for i in range(len(M_verts_faces))]
#%%
"""
# sizes = [x.vertices.shape[0] for x in M]
# if not os.path.exists('pai_matrices.pkl'):
# 	Adj = get_adj(A)
# 	bU = []
# 	bD = []
# 	for i in range(len(D)):
# 		d = np.zeros((1,D[i].shape[0]+1,D[i].shape[1]+1))
# 		u = np.zeros((1,U[i].shape[0]+1,U[i].shape[1]+1))
# 		d[0,:-1,:-1] = D[i].todense()
# 		u[0,:-1,:-1] = U[i].todense()
# 		d[0,-1,-1] = 1
# 		u[0,-1,-1] = 1
# 		bD.append(d)
# 		bU.append(u)
# 	bD = [sp.csr_matrix(s[0, ...]) for s in bD]
# 	bU = [sp.csr_matrix(s[0, ...]) for s in bU]
# 	with open('pai_matrices.pkl', 'wb') as fp:
# 		pickle.dump([Adj, sizes, bD, bU], fp)
# else:
"""
print("Loading adj Matrices ..")
with open('demoModel/CoMAdataset/PAIConv/pai_matrices.pkl', 'rb') as fp:
	[Adj, sizes, bD, bU] = pickle.load(fp)

tD = [sparse_mx_to_torch_sparse_tensor(s) for s in bD]
tU = [sparse_mx_to_torch_sparse_tensor(s) for s in bU]

#%%
model = PaiAutoencoder(filters_enc=filter_sizes_enc,
                       filters_dec=filter_sizes_dec,
                       latent_size=nz,
                       sizes=sizes,
                       t_vertices=vertices,
                       num_neighbors=kernal_size,
                       x_neighbors=Adj,
                       D=tD, U=tU).to(device)
model = torch.nn.DataParallel(model)
trainables_wo_index = [param for name, param in model.named_parameters()
                if param.requires_grad and 'adjweight' not in name]
trainables_wt_index = [param for name, param in model.named_parameters()
                if param.requires_grad and 'adjweight' in name]
optimizer = torch.optim.Adam([{'params': trainables_wo_index, 'weight_decay': regularization},
                          {'params': trainables_wt_index}],
                          lr=lr)

#
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            args.decay_step,
                                            gamma=args.lr_decay)

train_eval.run(model, train_loader, test_loader, args.epochs, optimizer,
               scheduler, writer, device)
#
# train_eval.eval_error(model, test_loader, device, meshdata, args.out_dir)
