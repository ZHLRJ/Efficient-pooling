# -*- coding: utf-8 -*-
'''
@Time    : 4/24/22 
@Author  : Zhang Haoliang
'''
from datasets import dfaust
from torch import Tensor
import numpy as np
import torch
class dfaustmeshdata(object):
    def __init__(self,root):
        self.root = root
        self.train_dataset = None
        self.test_dataset  = None
        self.test_type     = None
        self.num_train_graph = None
        self.num_test_graph  = None
        self.traindata=None
        self.testdata=None
        self.mean = None
        self.std=None
        self.load()

    def load(self):
        Data=dfaust.DFAUST(self.root).process()
        self.train_dataset = Data[0]
        self.test_dataset = Data[1]
        self.test_type      =Data[2]
        self.template_face = Data[3]
        self.num_nodes = self.train_dataset[0].num_nodes

        self.num_train_graph = len(self.train_dataset)
        self.num_test_graph = len(self.test_dataset)

        self.traindata=[data.x.unsqueeze(0) for data in self.train_dataset]
        self.traindata=torch.stack(self.traindata,0)

        self.testdata = [data.x.unsqueeze(0) for data in self.test_dataset]
        self.testdata = torch.stack(self.testdata, 0)
        self.mean=self.traindata.view(self.num_train_graph, -1,3).mean(dim=0)
        self.std = self.traindata.view(self.num_train_graph, -1,3).std(dim=0)

        # old stype require 30GB, terrible torch data structure
        # self.mean = self.train_dataset.data.x.view(self.num_train_graph, -1,
        #                                            3).mean(dim=0)
        #
        # self.traindata=self.train_dataset.data.x.view(self.num_train_graph, -1,
        #                                            3)
        #
        # self.std = self.train_dataset.data.x.view(self.num_train_graph, -1,
        #                                           3).std(dim=0)

        self.normalize()
    def normalize(self):
        print('Normalizing...')

        self.traindata = (self.traindata.view(self.num_train_graph, -1, 3) -
             self.mean) / self.std
        self.testdata = (self.testdata.view(self.num_test_graph, -1, 3) -
             self.mean) / self.std
        # self.traindata = self.traindata.view(self.num_train_graph, -1, 3)
        # self.testdata = self.testdata.view(self.num_test_graph, -1, 3)
        # print('Done!')
