# -*- coding: utf-8 -*-
'''
@Time    : 10/9/22 
@Author  : Zhang Haoliang
This File Pick some mesh from test dataset, used in paper to visualization
'''

# -*- coding: utf-8 -*-
'''
@Time    : 3/20/22 
@Author  : Zhang Haoliang
'''
import numpy as np
import trimesh
from torch_geometric.io import read_ply
# from result.trainedmodel.loadmodel import loadmodel
from vedo.utils import buildPolyData
from utils.writeOBJ import writeOBJ
import torch
from utils.loadPaperData import loadPaperData

# file="paperDemoData/headdata/groundTruth"
file="paperDemoData/dfaustdata/groundtruth"
#      file="paperDemoData/dfaustdata/groundtruth"
DataEx,face=loadPaperData(file)
# Plot
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# meshviewer(DataEx,FacesEX)
# Load model
from LoadTrainedModel_COMA_head import loadmodel
# model,mean,std=loadmodel(transform_fp="data/CoMA/transform.pkl",savedmodel="models/checkpoint_180.pt")
model,mean,std=loadmodel(transform_fp="data/CoMA/transform.pkl",savedmodel="comatrainedmodel/checkpoint_180.pt")

groundTruth=DataEx.numpy()
# mean, std = torch.load('comameanstd.pt')

x = ((DataEx -mean) /std)
# x=DataEx
errors = []
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    num_graphs = len(x)
    reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
    reshaped_x = (x.view(num_graphs, -1, 3).cpu() * std) + mean


    reshaped_pred *= 1000
    reshaped_x *= 1000
    mesh_error = torch.sqrt(
        torch.sum((reshaped_pred - reshaped_x) ** 2,
                  dim=2))  # [num_graphs, num_nodes]
    colorerror = torch.sqrt(abs(reshaped_pred - reshaped_x))
    errors.append(mesh_error)
new_errors = torch.cat(errors, dim=0)  # [n_total_graphs, num_nodes]

mean_error = new_errors.view((-1, )).mean()
std_error = new_errors.view((-1, )).std()
median_error = new_errors.view((-1, )).median()

message = 'Error: {:.3f}+{:.3f} | {:.3f}'.format(mean_error, std_error,
                                             median_error)
print(message)
# Generate error mesh
predArray=np.array(reshaped_pred)
for i in range(9):
    fileSaveName="./paperDemoData/headdata/pred/"+str(i+1)+"_"+"comaPred.obj"
    vpred=predArray[i]
    writeOBJ(fileSaveName,vpred/1000,face)

def rgb(value,minimum=0, maximum=255):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b

vertexcolor = []
# The error values are saturated at 7.5% of the geodesic diameter
maxerror=6.5
errorofpred=[]
for error in mesh_error:
    # error=np.array(error)
    error[error>maxerror]=maxerror
    # error/=maxerror
    colormap=[rgb(error[i],minimum=0, maximum=maxerror) for i in range(len(error))]
    vertexcolor.append(colormap)
    errorofpred.append(error)
    # break
# img_name='CoMAerrorbar.png',save=True,
from vedo import trimesh2vedo, show,Plotter,Mesh
from vedo.utils import buildPolyData
from utils.visualizationMesh import meshviewer,errormesh
errormesh(DataEx,FacesEX,errorofpred,scalarbar=False,img_name="spiral_new2.png",save=True)

# meshes = buildPolyData(DataEx, FacesEX)
# meshes = Mesh(meshes)
#
# meshes.cmap('jet', errorofpred, vmin=0, vmax=maxerror).addScalarBar(horizontal=True,size=(2500,190),titleFontSize=25)
#
# meshes.show()