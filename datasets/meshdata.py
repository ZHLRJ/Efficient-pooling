import openmesh as om
from datasets import CoMA
import os.path as  osp
import os
import numpy as np
class MeshData(object):
    def __init__(self,
                 root,
                 template_fp,
                 split='interpolation',
                 test_exp='bareteeth',
                 transform=None,
                 pre_transform=None):
        self.root = root
        self.template_fp = template_fp
        self.split = split
        self.test_exp = test_exp
        self.transform = transform
        self.pre_transform = pre_transform
        self.train_dataset = None
        self.test_dataste = None
        self.template_points = None
        self.template_face = None
        self.mean = None
        self.std = None
        self.num_nodes = None
        self.testfiles=[]

        self.load()

    def load(self):
        self.train_dataset = CoMA(self.root,
                                  train=True,
                                  split=self.split,
                                  test_exp=self.test_exp,
                                  transform=self.transform,
                                  pre_transform=self.pre_transform)
        self.test_dataset = CoMA(self.root,
                                 train=False,
                                 split=self.split,
                                 test_exp=self.test_exp,
                                 transform=self.transform,
                                 pre_transform=self.pre_transform)

        tmp_mesh = om.read_trimesh(self.template_fp)
        self.template_points = tmp_mesh.points()
        self.template_face = tmp_mesh.face_vertex_indices()
        self.num_nodes = self.train_dataset[0].num_nodes

        self.num_train_graph = len(self.train_dataset)
        self.num_test_graph = len(self.test_dataset)
        self.mean = self.train_dataset.data.x.view(self.num_train_graph, -1,
                                                   3).mean(dim=0)

        # self.train_dataset=self.train_dataset.data.x.view(self.num_train_graph, -1,
        #                                            3)
        # print(f"self.traindata.shape():{self.traindata.shape}")
        # plustemplate=np.vstack((self.traindata,self.template_points))

        self.std = self.train_dataset.data.x.view(self.num_train_graph, -1,
                                                  3).std(dim=0)
        self.loadtestfile()
        self.normalize()

    def normalize(self):
        print('Normalizing...')

        self.train_dataset.data.x = (
            (self.train_dataset.data.x.view(self.num_train_graph, -1, 3) -
             self.mean) / self.std).view(-1, 3)
        self.test_dataset.data.x = (
            (self.test_dataset.data.x.view(self.num_test_graph, -1, 3) -
             self.mean) / self.std).view(-1, 3)
        print('Done!')

    def save_mesh(self, fp, x):
        x = x * self.std + self.mean
        om.write_mesh(fp, om.TriMesh(x.numpy(), self.template_face))
    def loadtestfile(self,trainSave=False):
        if not trainSave:
            if self.split=='interpolation':
                logname=f"test-{self.split}.txt"
            elif self.split=='extrapolation':
                logname = f"test-{self.split}-{self.test_exp}.txt"
        path=osp.join(self.root,logname)
        if os.path.exists(path):
            with open(path,'r') as logfile:
                for line in logfile:
                    self.testfiles.append(line.strip('\n'))