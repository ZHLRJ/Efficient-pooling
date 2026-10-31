# -*- coding: utf-8 -*-
'''
@Time    : 4/24/22 
@Author  : Zhang Haoliang
'''
from utils import area_qslim
from psbody.mesh import Mesh
import trimesh
import pickle
from utils.visualizationMesh import errormesh,drawDownsampleMesh,upsampleerror

template_dfaust="template/dfaust/template.obj"
tdmesh=trimesh.load(template_dfaust)
mesh=Mesh(v=tdmesh.vertices,f=tdmesh.faces)
# mesh.show()
# tdmesh=trimesh.Trimesh(vertices=dmesh.v,faces=dmesh.f)
# tdmesh.show()
ds_factors=[4,4,4,4]
M, A, D, U, F=area_qslim.generate_transform_matrices(mesh, ds_factors)
meshes=[[mesh.v,mesh.f] for mesh in M]
transform_fp = "template/dfaust/transform_dfaust_new.pkl"

tmp = {'mesh':meshes,'face': F, 'adj': A, 'down_transform': D, 'up_transform': U}
with open(transform_fp, 'wb') as fp:
    pickle.dump(tmp, fp)
print('Done!')

transform_fp="template/dfaust/transform_dfaust_new.pkl"
with open(transform_fp, 'rb') as f:
    tmp = pickle.load(f, encoding='latin1')

M=tmp['mesh']



def draw(M):
    mesh_x,mesh_faces=[],[]
    for  mesh in M:

        mesh_x.append(mesh[0])
        mesh_faces.append(mesh[1])
    # "groundtruch_unpooling.png" "new_unpooling.png"
    # meshviewer(mesh_x,mesh_faces,img_name='down3.png',save=True,interval = 0.15,gap=300)
    drawDownsampleMesh(mesh_x,mesh_faces,shift=1,scale=2,
                       img_name="transform_dfaust_new.png",
                       cutheight=550,cutwidth=50)

from psbody.mesh import Mesh
import trimesh

mesh=Mesh(filename="prepare_DUmatrix/headtemplate.obj")
tmeshvn=trimesh.load("prepare_DUmatrix/headtemplate.obj")
t_mesh = trimesh.Trimesh(vertices=mesh.v,faces=mesh.f)
t_mesh_v = t_mesh.vertices
t_mesh_vn = 0.005 * t_mesh.vertex_normals
t_mesh.show()