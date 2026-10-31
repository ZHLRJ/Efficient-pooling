# -*- coding: utf-8 -*-
'''
@Time    : 10/9/22 
@Author  : Zhang Haoliang
'''


import os.path as osp
from models.Chebnetworks import AE
import pickle
import torch
from utils import utils

def loadmodel(transform_fp="data/CoMA/transform.pkl",savedmodel="models/checkpoint_300coma.pt"):
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    edge_index_list = [utils.to_edge_index(adj).to(device) for adj in tmp['adj']]
    down_transform_list = [
        utils.to_sparse(down_transform).to(device)
        for down_transform in tmp['down_transform']
    ]
    up_transform_list = [
        utils.to_sparse(up_transform).to(device)
        for up_transform in tmp['up_transform']
    ]

    model= AE(3,[16, 16, 16, 32],8,
               edge_index_list,
               down_transform_list,
               up_transform_list,
               K=9).to(device)
    PATH=savedmodel
    checkpoint = torch.load(PATH,map_location=torch.device('cpu'))
    optimizer = torch.optim.SGD(model.parameters(),
                                    lr=8e-3,
                                    weight_decay=5e-4,
                                    momentum=0.9)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    model.eval()
    mean,std=torch.load('models/comameanstd.pt')

    return model,mean,std

















