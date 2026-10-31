# -*- coding: utf-8 -*-
'''
@Time    : 10/10/22 
@Author  : Zhang Haoliang
'''
import os
from utils.readOBJ import readOBJ
import torch
def loadPaperData(filePath):
    # filePath="./paperDemoData/headdata/"
    FileNameList=[]
    meshDataTensor=[]
    for objFile in os.listdir(filePath):
        if objFile.endswith('obj'):
            FileNameList.append(objFile)
    FileNameList.sort()
    for objFile in FileNameList:
        path=os.path.join(filePath,objFile)
        V,F=readOBJ(path)
        VTensor = torch.Tensor(V)
        meshDataTensor.append(VTensor)
    meshDataTensor=torch.stack(meshDataTensor,dim=0)
    return meshDataTensor,F