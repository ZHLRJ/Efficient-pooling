# -*- coding: utf-8 -*-
'''
@Time    : 10/10/22 
@Author  : Zhang Haoliang
'''
import numpy as  np
def writeColorPLY(fileName,V,F,Vcolor):


    V=np.array(V)
    Vcolor=np.array(Vcolor)
    F=np.array(F)
    f = open(fileName, 'w')
    # write header
    f.write('ply\nformat {} 1.0\n'.format('ascii'))
    f.write('element vertex {}\n'.format(V.shape[0]))
    f.write('property double x\n')
    f.write('property double y\n')
    f.write('property double z\n')

    f.write('property uchar red\n')
    f.write('property uchar green\n')
    f.write('property uchar blue\n')
    f.write('property uchar alpha\n')

    f.write('element face {}\n'.format(F.shape[0]))
    f.write('property list uchar int vertex_indices\n')
    f.write('end_header\n')
    # f.close()

    Vstring='{:.17f} {:.17f} {:.17f} {} {} {} 255\n'
    Fstring="3 {} {} {}\n"


    for ii in range(V.shape[0]):
        f.write(Vstring.format(V[ii, 0],V[ii, 1],V[ii, 2],Vcolor[ii, 0],Vcolor[ii, 1],Vcolor[ii, 2] )  )

    for ii in range(F.shape[0]):
        f.write(Fstring.format(F[ii,0],F[ii,1],F[ii,2])  )
    f.close()