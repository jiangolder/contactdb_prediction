from torch.utils.data import Dataset
import torch
import os
import pickle
from torchvision import transforms
import numpy as np
import obman_utils as utils
import time
from collections import defaultdict
import trimesh

class obman(Dataset):
    def __init__(self, img_root='/hand-object-3/download/dataset/ObMan/obman',
                 obj_root='/hand-object-3/download/dataset',
                 mode="train"):

        # self.mode = mode
        # if self.mode == "train":
        #     self.file_path = '/hand-object-3/download/dataset/ObMan/obman/train.txt'
        # elif self.mode == "val":
        #     self.file_path = '/hand-object-3/download/dataset/ObMan/obman/val.txt'
        # else:
        #     self.file_path = '/hand-object-3/download/dataset/ObMan/obman/test.txt'
        # self.img_list = utils.readTxt_obman(self.file_path)

        self.mesh_list = ['C:\\Users\\think\\server\\model_normalized.obj']
        #self.mesh_list = ['G:\\Hand-object\\object-FPN-ROI\\data\\HO3D_Object_models\\003_cracker_box\\points.xyz']
        self.img_list = ['G:\\obman\\train\\meta\\00000017.pkl']

        # self.img_root = img_root
        # self.obj_root = obj_root
        self.dataset_size = len(self.mesh_list)

        self.transform = transforms.ToTensor()
        self.sample_nPoint = 3000


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # # OBMAN quick test
        obj_mesh_path = self.mesh_list[idx]
        meta_path = self.img_list[idx]

        meta = pickle.load(open(meta_path, 'rb'))
        obj_scale = meta['obj_scale']

        obj_xyz_normalized = np.array(utils.fast_load_obj(open(obj_mesh_path))[0]['vertices'])
        obj_xyz_normalized = utils.pc_normalize(obj_xyz_normalized)
        obj_xyz_normalized = torch.tensor(obj_xyz_normalized, dtype=torch.float32)
        nPoint = obj_xyz_normalized.size(0)
        print(nPoint)
        obj_scale_tensor = torch.tensor(obj_scale).type_as(obj_xyz_normalized).repeat(nPoint, 1)
        obj_pc = torch.cat((obj_xyz_normalized, obj_scale_tensor), dim=-1).permute(1, 0)

        # YCB quick test
        # obj_mesh_path = self.mesh_list[idx]
        # obj_mesh = np.array(trimesh.load(obj_mesh_path).vertices).T
        # offset = (obj_mesh.max(1, keepdims=True) + obj_mesh.min(1, keepdims=True)) / 2
        # obj_mesh -= offset
        # scale = max(obj_mesh.max(1) - obj_mesh.min(1)) / 2
        # obj_mesh /= scale
        # obj_pc = np.vstack((obj_mesh, scale * np.ones(obj_mesh.shape[1])))
        # print(obj_pc.shape)
        # obj_pc = torch.tensor(obj_pc, dtype=torch.float32)

        # line = self.img_list[idx].strip()
        # img_path = os.path.join(self.img_root, self.mode, 'rgb_obj', line)
        # meta_path = img_path.replace('rgb_obj', 'meta').replace('jpg', 'pkl')
        # #img = Image.open(img_path)
        # meta = pickle.load(open(meta_path, 'rb'))

        # hand information
        # hand_pose = torch.tensor(meta['hand_pose'])
        # hand_shape = torch.tensor(meta['shape'])
        # hand_trans = torch.tensor(meta['trans'])
        # hand_info = torch.cat((hand_pose, hand_shape), dim=0) # [55]
        # hand_side = meta['side']
        #hand_xyz = torch.tensor(meta['verts_3d']).permute(1,0) # [778, 3] -> [3, 778]

        # object information
        # obj_id = meta['sample_id']
        #
        # obj_path = meta['obj_path']
        # obj_path_seg = obj_path.split('/')[4:]
        # obj_path_seg = [it + '/' for it in obj_path_seg]
        # obj_mesh_path = ''.join(obj_path_seg)[:-1]
        # obj_mesh_path = os.path.join(self.obj_root, obj_mesh_path)
        # obj_xyz_normalized = np.array(utils.fast_load_obj(open(obj_mesh_path))[0]['vertices']) # [N, 3]
        # obj_xyz_normalized = torch.tensor(obj_xyz_normalized, dtype=torch.float32)
        # nPoint = obj_xyz_normalized.size(0)
        #
        # obj_scale = meta['obj_scale']
        # obj_scale_tensor = torch.tensor(obj_scale).type_as(obj_xyz_normalized).repeat(nPoint, 1)  # [N, 1]
        # obj_pc = torch.cat((obj_xyz_normalized, obj_scale_tensor), dim=-1).permute(1, 0) # [4,N]

        return (obj_pc, torch.tensor(idx))

