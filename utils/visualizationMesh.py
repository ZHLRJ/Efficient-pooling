# -*- coding: utf-8 -*-
'''
@Time    : 3/15/22 
@Author  : Zhang Haoliang
'''
import trimesh
import os.path as osp
import numpy as np
from vedo import trimesh2vedo, show,Plotter,Mesh
from vedo.utils import buildPolyData
import imageio

def meshviewer(mesh_x,mesh_faces,vertex_colors=[[192, 192, 192]],angle_Yrotate = -5,img_name=None,save=False,
               interval = 0.1,gap=220):
    # mesh_x, mesh_faces=Vlist,Flist
    meshes_for_plot = []
    idx=0
    for vertices, faces in zip(mesh_x, mesh_faces):
        meshobj = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors[idx])
        meshes = trimesh2vedo(meshobj)
        # meshes.pointColors(vertex_colors[idx])
        meshes.rotateY(angle_Yrotate, locally=True)
        meshes_for_plot.append(meshes)
        if vertex_colors[0][0]!=192:
            idx+=1
    # visualizationMesh.visulMesh(meshes_for_plot,angle=10,sharecam=False)
    plt = Plotter(bg2='w', size=(gap * len(mesh_x), 500))
    interval = interval
    for idx in range(len(meshes_for_plot)):
        # add draggable inset windows:
        plt.addInset(meshes_for_plot[idx].lighting(diffuse=0.9, specularPower=80), pos=(idx * interval + interval, 0.5),
                     size=0.4)
    if save==False:
        plt.show(interactive=True)
    else:
        if img_name==None:
            print("Please name the image")
        else:
            plt.show(interactive=False)
            cutheight, cutwidth = 100, 100
            img = plt.screenshot(asarray=True)[cutheight:-cutheight, cutwidth:-cutwidth]
            # plt.show(img)
            imageio.imwrite(img_name, img)
def errormesh(verticeslist,faceslist,errorlist,scalarbar=False,maxerror=6.5,angle_Yrotate = -5,
              img_name=None,save=False,interval=0.1):
    meshes_for_plot = []

    for vertices, faces,scalar in zip(verticeslist,faceslist,errorlist):
        meshes = buildPolyData(vertices, faces)
        meshes = Mesh(meshes)

        meshes.cmap('jet', scalar, vmin=0, vmax=maxerror)

        meshes_for_plot.append(meshes)
    # 220 for 9 image
    plt = Plotter(bg2='w', size=(220 * len(verticeslist), 500))

    for idx in range(len(meshes_for_plot)):
        # add draggable inset windows:
        if scalarbar and idx==len(meshes_for_plot)-1:
            meshes_for_plot[idx].addScalarBar(horizontal=True, pos=(0.8, 0.5))


        plt.addInset(meshes_for_plot[idx].lighting(diffuse=0.9, specularPower=80),
                     pos=(idx * interval + interval, 0.5),
                     size=0.4)

    if save == False:
        plt.show(interactive=True)
    else:
        if img_name == None:
            print("Please name the image")
        else:
            plt.show(interactive=False)
            cutheight, cutwidth = 100, 100
            img = plt.screenshot(asarray=True)[cutheight:-cutheight, cutwidth:-cutwidth]
            # plt.show(img)
            imageio.imwrite(img_name, img)
def drawDownsampleMesh(vlist,flist,shift=0.4,scale=1,showstate=True,img_name="new.png",cutheight=150, cutwidth=10):
    plt = Plotter()
    idx=0
    if type(shift)!=list:
        shift=[[shift*idx,0,0] for idx in range(len(vlist))]
    else:
        shift = [[shift[idx], 0, 0] for idx in range(len(vlist))]
    for v,f in zip(vlist,flist):
        v+=shift[idx]
        vtkmesh = buildPolyData(v, f)
        vtkmesh = Mesh(vtkmesh)
        vtkmesh.scale(scale)
        vtkmesh.c([192, 192, 192])

        vtkwire = vtkmesh.clone()
        # vtkmesh.alpha(0.3)
        vtkwire.wireframe()
        vtkwire.c([255, 0, 0])
        # plt += vtkwire
        plt += vtkmesh
        idx+=1
    plt.show(interactive=showstate)
    if showstate==False:
    # cutheight, cutwidth = 100, 100
        img = plt.screenshot(asarray=True)[cutheight:-cutheight, cutwidth:-cutwidth]
        # plt.show(img)
        imageio.imwrite(img_name, img)


def upsampleerror(vlist,flist,errorlist,shift=0.4,scale=1,maxerror=6.5,showstate=True,
                  img_name="new.png",maptype="coolwarm",colorbar=False,pos=(0.5,0.5),
                  cutheight=150, cutwidth=10):
    plt = Plotter()
    idx=0
    if type(shift)!=list:
        shift=[[shift*idx,0,0] for idx in range(len(vlist))]
    else:
        shift = [[shift[idx], 0, 0] for idx in range(len(vlist))]

    for v,f,error in zip(vlist,flist,errorlist):
        v+=shift[idx]
        vtkmesh = buildPolyData(v, f)
        vtkmesh = Mesh(vtkmesh)
        vtkmesh.scale(scale)
        vtkmesh.cmap(maptype,error, vmin=0, vmax=maxerror)
        if idx==len(vlist)-1 and colorbar:
            vtkmesh.addScalarBar(pos=pos,horizontal=True,titleFontSize=18)
        plt += vtkmesh
        idx+=1

    plt.show(interactive=showstate)
    if showstate==False:
    # cutheight, cutwidth = 100, 100
        if colorbar:
            img = plt.screenshot(asarray=True)[cutheight+50:-cutheight, cutwidth:-cutwidth]
        else:
            img = plt.screenshot(asarray=True)[cutheight:-cutheight, cutwidth:-cutwidth]
        # plt.show(img)
        imageio.imwrite(img_name, img)
