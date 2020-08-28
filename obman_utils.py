import torch
import numpy as np
import os
import pickle

def idx2onehot(idx, n):

    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)

    return onehot

def readTxt_obman(file_path):
    img_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip()
            img_list.append(item)
    file_to_read.close()
    return img_list

def vertices_transformation(vertices, rt):
    p = np.matmul(rt[:3,0:3], vertices.T) + rt[:3,3].reshape(-1,1)
    return p.T

def vertices_rotation(vertices, rt):
    p = np.matmul(rt[:3,0:3], vertices.T)
    return p.T

def fast_load_obj(file_obj, **kwargs):
    """
    Code slightly adapted from trimesh (https://github.com/mikedh/trimesh)
    Thanks to Michael Dawson-Haggerty for this great library !
    loads an ascii wavefront obj file_obj into kwargs
    for the trimesh constructor.
    vertices with the same position but different normals or uvs
    are split into multiple vertices.
    colors are discarded.
    parameters
    ----------
    file_obj : file object
                   containing a wavefront file
    returns
    ----------
    loaded : dict
                kwargs for trimesh constructor
    """
    # make sure text is utf-8 with only \n newlines
    text = file_obj.read()
    if hasattr(text, 'decode'):
        text = text.decode('utf-8')
    text = text.replace('\r\n', '\n').replace('\r', '\n') + ' \n'
    meshes = []
    def append_mesh():
        # append kwargs for a trimesh constructor
        # to our list of meshes
        if len(current['f']) > 0:
            # get vertices as clean numpy array
            vertices = np.array(
                current['v'], dtype=np.float64).reshape((-1, 3))
            # do the same for faces
            faces = np.array(current['f'], dtype=np.int64).reshape((-1, 3))
            # get keys and values of remap as numpy arrays
            # we are going to try to preserve the order as
            # much as possible by sorting by remap key
            keys, values = (np.array(list(remap.keys())),
                            np.array(list(remap.values())))
            # new order of vertices
            vert_order = values[keys.argsort()]
            # we need to mask to preserve index relationship
            # between faces and vertices
            face_order = np.zeros(len(vertices), dtype=np.int64)
            face_order[vert_order] = np.arange(len(vertices), dtype=np.int64)
            # apply the ordering and put into kwarg dict
            loaded = {
                'vertices': vertices[vert_order],
                'faces': face_order[faces],
                'metadata': {}
            }
            # build face groups information
            # faces didn't move around so we don't have to reindex
            if len(current['g']) > 0:
                face_groups = np.zeros(len(current['f']) // 3, dtype=np.int64)
                for idx, start_f in current['g']:
                    face_groups[start_f:] = idx
                loaded['metadata']['face_groups'] = face_groups
            # we're done, append the loaded mesh kwarg dict
            meshes.append(loaded)
    attribs = {k: [] for k in ['v']}
    current = {k: [] for k in ['v', 'f', 'g']}
    # remap vertex indexes {str key: int index}
    remap = {}
    next_idx = 0
    group_idx = 0
    for line in text.split("\n"):
        line_split = line.strip().split()
        if len(line_split) < 2:
            continue
        if line_split[0] in attribs:
            # v, vt, or vn
            # vertex, vertex texture, or vertex normal
            # only parse 3 values, ignore colors
            attribs[line_split[0]].append([float(x) for x in line_split[1:4]])
        elif line_split[0] == 'f':
            # a face
            ft = line_split[1:]
            if len(ft) == 4:
                # hasty triangulation of quad
                ft = [ft[0], ft[1], ft[2], ft[2], ft[3], ft[0]]
            for f in ft:
                # loop through each vertex reference of a face
                # we are reshaping later into (n,3)
                if f not in remap:
                    remap[f] = next_idx
                    next_idx += 1
                    # faces are "vertex index"/"vertex texture"/"vertex normal"
                    # you are allowed to leave a value blank, which .split
                    # will handle by nicely maintaining the index
                    f_split = f.split('/')
                    current['v'].append(attribs['v'][int(f_split[0]) - 1])
                current['f'].append(remap[f])
        elif line_split[0] == 'o':
            # defining a new object
            append_mesh()
            # reset current to empty lists
            current = {k: [] for k in current.keys()}
            remap = {}
            next_idx = 0
            group_idx = 0
        elif line_split[0] == 'g':
            # defining a new group
            group_idx += 1
            current['g'].append((group_idx, len(current['f']) // 3))
    if next_idx > 0:
        append_mesh()
    return meshes

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def get_img_list_val(mode):
    if mode == "train":
        file_path = '/hand-object-3/hanwen/contactdb_prediction/logs/2020_8_24_9_train/log_unprocessed.txt'
        file_path = '/hand-object-3/download/dataset/ObMan/obman/train.txt'
    elif mode == "val":
        file_path = '/hand-object-3/hanwen/contactdb_prediction/logs/2020_8_24_7_val/log_unprocessed.txt'
        file_path = '/hand-object-3/download/dataset/ObMan/obman/val.txt'
    else:
        file_path = '/hand-object-3/hanwen/contactdb_prediction/logs/2020_8_24_8_test/log_unprocessed.txt'
        file_path = '/hand-object-3/download/dataset/ObMan/obman/test.txt'
    img_list = readTxt_obman(file_path)
    return img_list

def get_saveName(line, mode):
    img_root='/hand-object-3/download/dataset/ObMan/obman'
    obj_root = '/hand-object-3/download/dataset'

    line = line.strip()
    img_path = os.path.join(img_root, mode, 'rgb_obj', line)
    meta_path = img_path.replace('rgb_obj', 'meta').replace('jpg', 'pkl')
    meta = pickle.load(open(meta_path, 'rb'))

    obj_path = meta['obj_path']
    obj_path_seg = obj_path.split('/')[4:]
    obj_path_seg = [it + '/' for it in obj_path_seg]
    obj_mesh_path = ''.join(obj_path_seg)[:-1]
    obj_mesh_path = os.path.join(obj_root, obj_mesh_path)
    save_name = obj_mesh_path.replace('model_normalized', 'contactmap2').replace('.obj', '.npy')
    return save_name