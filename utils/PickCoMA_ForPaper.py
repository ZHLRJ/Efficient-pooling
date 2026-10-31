# -*- coding: utf-8 -*-
'''
@Time    : 10/10/22
@Author  : Zhang Haoliang

Save 9 different OBJ CoMA data for paper use
Save to "./paperDemoData/headdata/"+str(idx+1)+"_"+catg+".obj"
'''
import numpy as np
import trimesh
from torch_geometric.io import read_ply
# from result.trainedmodel.loadmodel import loadmodel
from vedo.utils import buildPolyData
from utils.writeOBJ import writeOBJ
import torch


"""
Pick CoMA data
"""

categories = ['bareteeth','cheeks_in','eyebrow','high_smile','lips_back','lips_up','mouth_down',
              'mouth_extreme','mouth_middle','mouth_open','mouth_side','mouth_up',]
# Pick 9 different face
pickedcat=['cheeks_in','eyebrow','high_smile','lips_up','mouth_down',
              'mouth_extreme','mouth_open','mouth_side','mouth_up',]
testInterpolationFile="data/CoMA/test-interpolation.txt"
# Save test data with categories
testFileInCategories={categories[i]:[] for i in range(len(categories))}
with open(testInterpolationFile,'r') as log:
    Allfiles=log.readlines()
    for file in Allfiles:
        file=file.strip("\n")
        categoryName=file.split('/')[-1].split('.')[0]
        testFileInCategories[categoryName].append(file)
CategoriesEx=[]
CategoriesExData=[]
VerticesEx=[]
FacesEX=[]
for idx,catg in enumerate(pickedcat) :
    pickFile=testFileInCategories[catg][0]
    CategoriesEx.append(pickFile)
    # trimeshdata=trimesh.load(pickFile)
    meshesfromply=read_ply(pickFile)
    plyvertex,faces=np.array(meshesfromply.pos),np.array(meshesfromply.face,dtype=int).transpose()
    #
    # fileSaveName="./paperDemoData/headdata/"+str(idx+1)+"_"+catg+".obj"
    # writeOBJ(fileSaveName,plyvertex,faces)
    plyvertexTensor=torch.Tensor(plyvertex)
    VerticesEx.append(plyvertexTensor)
    FacesEX.append(torch.Tensor(faces))
    CategoriesExData.append(plyvertex)
DataEx=torch.stack(VerticesEx,dim=0)
FacesEX=torch.stack(FacesEX,dim=0)
print(f"DataEx shape : {DataEx.shape}, FacesEx shape {FacesEX.shape}")


"""
Pick dfaust data
"""
from datasets import dfaustmeshdata
from utils.visualizationMesh import drawDownsampleMesh,upsampleerror
import numpy as np
import torch
root_dir="data/Dfaust"
meshdata = dfaustmeshdata(root_dir)

Data_test=meshdata.testdata
Test_type=meshdata.test_type
mean,std=meshdata.mean,meshdata.std
face=meshdata.template_face

for idx, i in enumerate([4899, 3230, 4567, 1223, 1237]):
    mesh_v = Data_test[i] * std + mean
    fileSaveName = "./paperDemoData/dfaustdata/" + str(idx + 1) + "_" + str(i) + ".obj"
    writeOBJ(fileSaveName, mesh_v.numpy(), face)

