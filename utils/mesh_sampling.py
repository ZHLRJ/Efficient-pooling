# -*- coding: utf-8 -*-
'''
@Time    : 4/16/22
@Author  : Zhang Haoliang
'''
import math
import heapq
import scipy.sparse as sp
import numpy as np
import os

from psbody.mesh import Mesh
from scipy.spatial import KDTree
import pickle

def row(A):
    return A.reshape((1, -1))


def col(A):
    return A.reshape((-1, 1))


def get_vert_connectivity(mesh_v, mesh_f):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12."""

    vpv = sp.csc_matrix((len(mesh_v), len(mesh_v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh_f[:, i]
        JS = mesh_f[:, (i + 1) % 3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.ravel()), row(JS.ravel())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv


def get_vertices_per_edge(mesh_v, mesh_f):
    """Returns an Ex2 array of adjacencies between vertices, where
    each element in the array is a vertex index. Each edge is included
    only once. If output of get_faces_per_edge is provided, this is used to
    avoid call to get_vert_connectivity()"""

    vc = sp.coo_matrix(get_vert_connectivity(mesh_v, mesh_f))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:, 0] < result[:, 1]]  # for uniqueness

    return result


def vertex_quadrics(mesh):
    """Computes a quadric for each vertex in the Mesh.

    Returns:
       v_quadrics: an (N x 4 x 4) array, where N is # vertices.
    """

    # Allocate quadrics
    v_quadrics = np.zeros((
        len(mesh.v),
        4,
        4,
    ))

    # For each face...
    for f_idx in range(len(mesh.f)):

        # Compute normalized plane equation for that face
        vert_idxs = mesh.f[f_idx]
        verts = np.hstack((mesh.v[vert_idxs], np.array([1, 1,
                                                        1]).reshape(-1, 1)))
        u, s, v = np.linalg.svd(verts)
        eq = v[-1, :].reshape(-1, 1)
        eq = eq / (np.linalg.norm(eq[0:3]))

        # Add the outer product of the plane equation to the
        # quadrics of the vertices for this face
        for k in range(3):
            v_quadrics[mesh.f[f_idx, k], :, :] += np.outer(eq, eq)

    return v_quadrics


def setup_deformation_transfer(source, target, use_normals=False):
    rows = np.zeros(3 * target.v.shape[0])
    cols = np.zeros(3 * target.v.shape[0])
    coeffs_v = np.zeros(3 * target.v.shape[0])
    coeffs_n = np.zeros(3 * target.v.shape[0])

    nearest_faces, nearest_parts, nearest_vertices = source.compute_aabb_tree(
    ).nearest(target.v, True)
    nearest_faces = nearest_faces.ravel().astype(np.int64)
    nearest_parts = nearest_parts.ravel().astype(np.int64)
    nearest_vertices = nearest_vertices.ravel()

    for i in range(target.v.shape[0]):
        # Closest triangle index
        # i=0
        f_id = nearest_faces[i]
        # Closest triangle vertex ids
        nearest_f = source.f[f_id]

        # Closest surface point
        nearest_v = nearest_vertices[3 * i:3 * i + 3]
        # Distance vector to the closest surface point
        dist_vec = target.v[i] - nearest_v

        rows[3 * i:3 * i + 3] = i * np.ones(3)
        cols[3 * i:3 * i + 3] = nearest_f

        n_id = nearest_parts[i]
        # newmethods
        V=source.v[nearest_f][0]
        v1, v2, v3 = V[0],V[1],V[2]
        vector_1, vector_2 = v2 - v1, v3 - v1
        vector_1/=np.linalg.norm(vector_1)
        vector_2/=np.linalg.norm(vector_2)
        perpendicular = np.cross(vector_1, vector_2)
        perpendicular/=np.linalg.norm(perpendicular)
        coord_sys=np.vstack((vector_1,vector_2,perpendicular)).T
        target=mesh.v[to_destroy]
        # coeffs_v=np.linalg.lstsq(coord_sys, target,rcond=-1)[0]
        ###############
        if n_id == 0:
            # Closest surface point in triangle
            A = np.vstack((source.v[nearest_f])).T
            coeffs_v[3 * i:3 * i + 3] = np.linalg.lstsq(A, nearest_v,
                                                        rcond=-1)[0]
        elif n_id > 0 and n_id <= 3:
            # Closest surface point on edge
            A = np.vstack((source.v[nearest_f[n_id - 1]],
                           source.v[nearest_f[n_id % 3]])).T
            tmp_coeffs = np.linalg.lstsq(A, target.v[i], rcond=-1)[0]
            coeffs_v[3 * i + n_id - 1] = tmp_coeffs[0]
            coeffs_v[3 * i + n_id % 3] = tmp_coeffs[1]
        else:
            # Closest surface point a vertex
            coeffs_v[3 * i + n_id - 4] = 1.0

    matrix = sp.csc_matrix((coeffs_v, (rows, cols)),
                           shape=(target.v.shape[0], source.v.shape[0]))
    return matrix
def new_setup_deformation_transfer(source, target, use_normals=False):
    source, target=M[-1], M[-2]
    rows = np.zeros(3 * target.v.shape[0])
    cols = np.zeros(3 * target.v.shape[0])
    # coeffs_v = np.zeros(3 * target.v.shape[0])
    # coeffs_n = np.zeros(3 * target.v.shape[0])

    nearest_faces, nearest_parts, nearest_vertices = source.compute_aabb_tree(
    ).nearest(target.v, True)
    nearest_faces = nearest_faces.ravel().astype(np.int64)
    nearest_parts = nearest_parts.ravel().astype(np.int64)
    nearest_vertices = nearest_vertices.ravel()
    Upinfo=[]
    specialcase=[]
    for i in range(target.v.shape[0]):
        # Closest triangle index
        # i=0
        f_id = nearest_faces[i]
        # Closest triangle vertex ids
        nearest_f = source.f[f_id]

        # Closest surface point
        nearest_v = nearest_vertices[3 * i:3 * i + 3]
        # Distance vector to the closest surface point
        dist_vec = target.v[i] - nearest_v
        target_vertex = target.v[i]
        rows[3 * i:3 * i + 3] = i * np.ones(3)
        cols[3 * i:3 * i + 3] = nearest_f

        n_id = nearest_parts[i]
        ###############
        if 0<=n_id and n_id <= 3:
            # Closest surface point in triangle
            # newmethods
            V = source.v[nearest_f]
            v1, v2, v3 = V[0], V[1], V[2]
            vector_1, vector_2 = v2 - v1, v3 - v1
            vector_1 /= np.linalg.norm(vector_1)
            vector_2 /= np.linalg.norm(vector_2)
            perpendicular = np.cross(vector_1, vector_2)
            perpendicular /= np.linalg.norm(perpendicular)
            coord_sys = np.vstack((vector_1, vector_2, perpendicular)).T
            weight = np.linalg.lstsq(coord_sys, target_vertex, rcond=-1)[0]

            # coeffs_v[3 * i:3 * i + 3] = weight
            Upinfo.append([nearest_f,weight])
        else:
            # Closest surface point a vertex
            # coeffs_v[3 * i + n_id - 4] = 1.0
            # print(n_id)
            Upinfo.append([nearest_f[n_id - 4]])
            # specialcase.append(i)

    # matrix = sp.csc_matrix((coeffs_v, (rows, cols)),
    #                        shape=(target.v.shape[0], source.v.shape[0]))
    return Upinfo
def _get_sparse_transform(faces, num_original_verts):
    verts_left = np.unique(faces.flatten())
    IS = np.arange(len(verts_left))
    JS = verts_left
    data = np.ones(len(JS))

    mp = np.arange(0, np.max(faces.flatten()) + 1)
    mp[JS] = IS
    new_faces = mp[faces.copy().flatten()].reshape((-1, 3))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij),
                        shape=(len(verts_left), num_original_verts))

    return (new_faces, mtx)

def collapse_cost(Qv, r, c, v):
    Qsum = Qv[r, :, :] + Qv[c, :, :]
    p1 = np.vstack((v[r].reshape(-1, 1), np.array([1]).reshape(-1, 1)))
    p2 = np.vstack((v[c].reshape(-1, 1), np.array([1]).reshape(-1, 1)))

    destroy_c_cost = p1.T.dot(Qsum).dot(p1)
    destroy_r_cost = p2.T.dot(Qsum).dot(p2)
    result = {
        'destroy_c_cost': destroy_c_cost,
        'destroy_r_cost': destroy_r_cost,
        'collapse_cost': min([destroy_c_cost, destroy_r_cost]),
        'Qsum': Qsum
    }
    return result

def edgecosts_queue(Qv,mesh):

    # construct a queue of edges with costs
    vert_adj = get_vertices_per_edge(mesh.v, mesh.f)
    vert_adj = sp.csc_matrix(
        (vert_adj[:, 0] * 0 + 1, (vert_adj[:, 0], vert_adj[:, 1])),
        shape=(len(mesh.v), len(mesh.v)))
    vert_adj = vert_adj + vert_adj.T
    vert_adj = vert_adj.tocoo()
    # construct a queue of edges with costs
    cost_queue = []
    for k in range(vert_adj.nnz):
        r = vert_adj.row[k]
        c = vert_adj.col[k]

        if r > c:
            continue

        cost = collapse_cost(Qv, r, c, mesh.v)['collapse_cost']
        heapq.heappush(cost_queue, (cost, (r, c)))
    return cost_queue
def logical_or3(x, y, z):
    return np.logical_or(x, np.logical_or(y, z))
def remove_degenerate_faces(faces):
    a = faces[:, 0] == faces[:, 1]
    b = faces[:, 1] == faces[:, 2]
    c = faces[:, 2] == faces[:, 0]

    # remove degenerate faces

    faces_to_keep = np.logical_not(logical_or3(a, b, c))
    faces = faces[faces_to_keep, :].copy()
    return faces
def update_queue(queue,to_destroy,to_keep):
    which1 = [
        idx for idx in range(len(queue))
        if queue[idx][1][0] == to_destroy
    ]
    which2 = [
        idx for idx in range(len(queue))
        if queue[idx][1][1] == to_destroy
    ]
    for k in which1:
        queue[k] = (queue[k][0], (to_keep, queue[k][1][1]))
    for k in which2:
        queue[k] = (queue[k][0], (queue[k][1][0], to_keep))
    return queue

def qslim_decimator_transformer(mesh, factor=0.25, n_verts_desired=None):
    if factor is None and n_verts_desired is None:
        raise Exception('Need either factor or n_verts_desired.')

    if n_verts_desired is None:
        n_verts_desired = math.ceil(len(mesh.v) * factor)
    Qv = vertex_quadrics(mesh)
    queue=edgecosts_queue(Qv,mesh)
    # decimate
    collapse_list = []
    nverts_total = len(mesh.v)
    faces = mesh.f.copy()

    while nverts_total > n_verts_desired:
    # runonce=0
    # while runonce<1:
    #     runonce+=1
        e = heapq.heappop(queue)
        # e=queue[-2800]
        # two vertices
        r ,c = e[1][0],e[1][1]
        if r == c:
            continue
        cost = collapse_cost(Qv, r, c, mesh.v)
        if cost['collapse_cost'] > e[0]:
            heapq.heappush(queue, (cost['collapse_cost'], e[1]))
            # print 'found outdated cost, %.2f < %.2f' % (e[0], cost['collapse_cost'])
            continue
        else:
            # update old vert idxs to new one,
            # in queue and in face list
            if cost['destroy_c_cost'] < cost['destroy_r_cost']:
                to_destroy = c
                to_keep = r
            else:
                to_destroy = r
                to_keep = c

            collapse_list.append([to_keep, to_destroy])

            # in our face array, replace "to_destroy" vertidx with "to_keep" vertidx
            np.place(faces, faces == to_destroy, to_keep)

            # same for queue
            queue=update_queue(queue, to_destroy,to_keep)
            Qv[r, :, :] = cost['Qsum']
            Qv[c, :, :] = cost['Qsum']
            # newfaces=remove_degenerate_faces(faces)
            faces=remove_degenerate_faces(faces)
            nverts_total = (len(np.unique(faces.flatten())))
        #
        # source=Mesh(v=mesh.v,f=faces)
        # nearest_faces, nearest_parts, nearest_vertices = source.compute_aabb_tree(
        # ).nearest([mesh.v[to_destroy]], True)

    new_faces, mtx = _get_sparse_transform(faces, len(mesh.v))
    return new_faces, mtx
def generate_transform_matrices(mesh, factors):
    """Generates len(factors) meshes, each of them is scaled by factors[i] and
       computes the transformations between them.

    Returns:
       M: a set of meshes downsampled from mesh by a factor specified in factors.
       A: Adjacency matrix for each of the meshes
       D: csc_matrix Downsampling transforms between each of the meshes
       U: Upsampling transforms between each of the meshes
       F: a list of faces
    """

    factors = map(lambda x: 1.0 / x, factors)
    M, A, D, U, F = [], [], [], [], []
    F.append(mesh.f)  # F[0]
    A.append(get_vert_connectivity(mesh.v, mesh.f).astype('float32'))  # A[0]
    M.append(mesh)  # M[0]

    for factor in factors:
        ds_f, ds_D = qslim_decimator_transformer(M[-1], factor=factor)
        D.append(ds_D.astype('float32'))
        new_mesh_v = ds_D.dot(M[-1].v)
        new_mesh = Mesh(v=new_mesh_v, f=ds_f)
        F.append(new_mesh.f)
        M.append(new_mesh)
        A.append(
            get_vert_connectivity(new_mesh.v, new_mesh.f).astype('float32'))
        U.append(setup_deformation_transfer(M[-1], M[-2]).astype('float32'))

    return M, A, D, U, F

# filename="template/template.obj"

mesh=Mesh(filename="template/template.obj")
M, A, D, U, F = [], [], [], [], []

# import pickle
M=[mesh]
ds_f, ds_D = qslim_decimator_transformer(mesh)
D.append(ds_D.astype('float32'))
new_mesh_v = ds_D.dot(M[-1].v)
new_mesh = Mesh(v=new_mesh_v, f=ds_f)
F.append(new_mesh.f)
M.append(new_mesh)
A.append(
    get_vert_connectivity(new_mesh.v, new_mesh.f).astype('float32'))
# U.append(new_setup_deformation_transfer(M[-1], M[-2]).astype('float32'))
U,upinfo=new_setup_deformation_transfer(M[-1], M[-2])
mesh=M[-1]
# U=U.toarray()
mesh_v=mesh.v
def mapback(upinfo,mesh_v):
    vertices=[]
    for i,item in enumerate(upinfo):
        if len(item)==1:
            vertices.append(mesh_v[item[0]])
        else:
            idx, weight=item
            V = mesh_v[idx]
            v1, v2, v3 = V[0], V[1], V[2]
            vector_1, vector_2 = v2 - v1, v3 - v1
            vector_1 /= np.linalg.norm(vector_1)
            vector_2 /= np.linalg.norm(vector_2)
            perpendicular = np.cross(vector_1, vector_2)
            perpendicular /= np.linalg.norm(perpendicular)
            coord_sys = np.vstack((vector_1, vector_2, perpendicular)).T
            coor=coord_sys.dot(weight)
            vertices.append(coor)
    return vertices
# vertices=mapback(upinfo,mesh.v)
# faces=M[-2].f
# mesh=Mesh(vertices,faces)
# from vedo import trimesh2vedo, show,Plotter,Points
# from vedo.utils import buildPolyData
# import vedo
# import trimesh
#
# vtkmesh = buildPolyData(mesh.v, mesh.f)
# vtkmesh = vedo.Mesh(vtkmesh)
# vtkmesh.c([0,0,0])
# vtkmesh.wireframe()
# plt = Plotter()
# plt+=vtkmesh
# # targetface=mesh.v[newfaces[nearest_faces[0][0]]]
# # plt += Points([mesh.v[to_destroy]], c=[255,0,0],r=8)
# # plt+=Points(mesh.v[newfaces[nearest_faces[0][0]]],c=[255,0,255],r=10)
# # plt+=Points(nearest_vertices,c=[0,255,0],r=10)
# plt.show()
#
