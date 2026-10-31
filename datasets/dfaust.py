# -*- coding: utf-8 -*-
'''
@Time    : 4/24/22 
@Author  : Zhang Haoliang
'''
import h5py
import os.path as osp
import numpy as np
import torch
from torch_geometric.data.collate import collate
from torch_geometric.data import Data as geo_data
from torch_geometric.utils import to_undirected
from tqdm import tqdm

def edge_extract(face):
    face = torch.from_numpy(np.array(face)).T.type(torch.long)
    edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    edge_index = to_undirected(edge_index)
    return edge_index


class DFAUST():

    def __init__(self,root_dir):
        self.root=root_dir
        self.raw_file=[osp.join(self.root,file) for file in self.raw_file_names()]
        self.faces=None
        # print(self.raw_file)
    def collate(self,data_list):

        if len(data_list) == 1:
            return data_list[0], None

        data, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,
        )

        return data, slices

    def raw_file_names(self):
        return ["registrations_f.hdf5","registrations_m.hdf5"]
    def process(self):
        train_test_file=osp.join(self.root,"train_test_v1.pt")
        if osp.exists(train_test_file):
            print(" train_test.pt exist, Loading... ")
            Data=torch.load(train_test_file)
            return Data["train"],Data["test"],Data["testtype"],Data["face"]
        RAW_DATA=[]
        TypeofMesh=[]
        for hdf5_file in self.raw_file:
            raw_data=h5py.File(hdf5_file, 'r')
            self.faces=np.array(raw_data['faces'])
            edge_index=edge_extract(raw_data['faces'])
            print(f"Load {raw_data.filename.split('/')[-1]} :")
            for key in tqdm(raw_data.keys()):
                if key!='faces':
                    num_of_data = raw_data[key].shape[-1]
                    TypeofMesh+=[key]*num_of_data
                    type_data=np.array(raw_data[key])

                    for i in range(num_of_data):
                        x=torch.tensor(type_data[...,i].astype('float32'))
                        data=geo_data(x=x, edge_index=edge_index)
                        RAW_DATA.append(data)
        train_dataset, test_dataset,test_type =[],[],[]
        N=len(RAW_DATA)
        shuffleidx = np.arange(N)
        np.random.shuffle(shuffleidx)
        testset = set(shuffleidx[-5000:])
        for i in range(N):
            if i in testset:
                test_dataset.append(RAW_DATA[i])
                test_type.append(TypeofMesh[i])
            else:
                train_dataset.append(RAW_DATA[i])
        StoreData={}
        # StoreData['train']=self.collate(train_dataset)
        # StoreData['test']=self.collate(test_dataset)
        StoreData['train'] = train_dataset
        StoreData['test'] = test_dataset
        StoreData['testtype']=test_type
        StoreData['face']=self.faces
        torch.save(StoreData,'train_test_v2.pt')
        return train_dataset, test_dataset,test_type,self.faces

