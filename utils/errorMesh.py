# -*- coding: utf-8 -*-
'''
@Time    : 10/10/22 
@Author  : Zhang Haoliang
'''
from utils.loadPaperData import loadPaperData
import torch
from utils.errorColorMap import errorColorMap
from utils.writeColorPLY import writeColorPLY
groudTruth, F = loadPaperData("paperDemoData/headdata/groundTruth")
outputMesh, _=loadPaperData("paperDemoData/headdata/pred3")

groudTruth*=1000
outputMesh*=1000
def errorMesh(groudTruth,outputMesh,F,errorThreshold=3.5):
    # errorThreshold = 3
    mesh_error = torch.sqrt(
        torch.sum((outputMesh - groudTruth) ** 2,
                  dim=2))  # [num_graphs, num_nodes]
    mesh_error[mesh_error>errorThreshold]=errorThreshold

    for idx,errorList in enumerate(mesh_error):
        colorMesh=errorColorMap(errorList,maxError=errorThreshold,useMatplot=True)
        V=outputMesh[idx]
        writeColorPLY("./paperDemoData/headdata/"+str(idx+1)+"colorPred"+".ply",V/1000,F,colorMesh)

errorMesh(groudTruth,outputMesh,F,errorThreshold=4)