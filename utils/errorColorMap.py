# -*- coding: utf-8 -*-
'''
@Time    : 10/10/22 
@Author  : Zhang Haoliang
'''
from utils.MyColorMap import myColorMap
import matplotlib
import numpy as np
CMap=myColorMap()
CMap=CMap[::-1]
# color map red-->blue 0-->maxError
MatplotCMap = matplotlib.cm.get_cmap("coolwarm")
def errorColorMap(errorList,maxError=3.5,CMap=CMap,useMatplot=False):

    errorRGB=[]
    N=len(CMap)-1
    for error in errorList:
        error=(error)/maxError
        if useMatplot:
            rgba=np.array(MatplotCMap(error.numpy()) )*255
            errorRGB.append([int(rgba[0]),int(rgba[1]),int(rgba[2])]  )
        else:
            errorRGB.append(list(CMap[int(error*N)]))
    return errorRGB