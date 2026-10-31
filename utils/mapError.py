# -*- coding: utf-8 -*-
'''
@Time    : 10/22/22 
@Author  : Zhang Haoliang
'''
import pickle
import trimesh
import numpy as np
from utils.writeOBJ import writeOBJ
transform_fp="demoModel/CoMAdataset/CoMAmodel/transform.pkl"
with open(transform_fp, 'rb') as f:
	tmp = pickle.load(f, encoding='latin1')
down_transform_list = [down_transform.toarray() for down_transform in tmp['down_transform']]
up_transform_list = [up_transform.toarray() for up_transform in tmp['up_transform']]
face_list=[face for face in tmp['face']]
template="template/CoMAtemplate.obj"
mesh=trimesh.load(template,process=False)


def generateUpsampling(vertices,downtransformlist,uptransform,level=1):
	for i in range(level):
		vertices = downtransformlist[i].dot(vertices)
	for i in range(level-1,-1,-1):
		vertices=uptransform[i].dot(vertices)
	return vertices


def meshError(examplePath,downtransformlist,uptransform,level=1):
	mesh = trimesh.load(examplePath, process=False)
	vertices=mesh.vertices
	upsampleVertices=generateUpsampling(vertices,downtransformlist,uptransform,level=level)
	errorMesh=upsampleVertices-vertices
	squareErrorMesh=np.square(errorMesh)*1000
	error=[]
	for i in range(squareErrorMesh.shape[0]):
		errordis=np.sqrt(sum(squareErrorMesh[i,...]))
		if errordis>0.01:
			error.append(errordis)
	return error

examplePath="data/CoMA/raw/COMA_data/FaceTalk_170731_00024_TA/mouth_extreme/mouth_extreme.000011.ply"
exampleError=meshError(examplePath,down_transform_list,up_transform_list,level=3)
templateError=meshError(template,down_transform_list,up_transform_list,level=3)
templateError=templateError[:2000]
exampleError=exampleError[:2000]

import matplotlib.pyplot as plt
def axcolor(ax, color='w'):
    ax.grid(color='w', axis="y", linewidth=2)
    ax.spines['left'].set_color(color)  # setting up Y-axis tick color
    ax.spines['top'].set_color(color)
    ax.spines['bottom'].set_color(color)  # setting up Y-axis tick color
    ax.spines['right'].set_color(color)

    ax.xaxis.set_tick_params(length=0, labelbottom=True)
    ax.yaxis.set_tick_params(length=0, labelbottom=True)

fig,ax=plt.subplots()
ax.set_facecolor((240/255,240/255,240/255))
axcolor(ax)
fig.set_size_inches(13, 6)
templateError=[min(0.2,i) for i in templateError]

ax.scatter(np.arange(len(exampleError)), exampleError, s=2, c='lightcoral', alpha=1)
ax.scatter(np.arange(len(templateError)), templateError, s=2, c='lightBlue', alpha=1)

plt.ylim([-0.1,0.5])
plt.savefig('ErrorDis.png')
plt.show()