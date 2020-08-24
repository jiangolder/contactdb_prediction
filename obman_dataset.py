from torch.utils.data import Dataset
import torch
import os
import pickle
from torchvision import transforms
import numpy as np
import obman_utils as utils
import time
from collections import defaultdict

class obman(Dataset):
    def __init__(self, img_root='/hand-object-3/download/dataset/ObMan/obman',
                 obj_root='/hand-object-3/download/dataset',
                 mode="train"):

        self.mode = mode
        if self.mode == "train":
            self.file_path = '/hand-object-3/download/dataset/ObMan/obman/train.txt'
        elif self.mode == "val":
            self.file_path = '/hand-object-3/download/dataset/ObMan/obman/val.txt'
        else:
            self.file_path = '/hand-object-3/download/dataset/ObMan/obman/test.txt'
        self.img_list = utils.readTxt_obman(self.file_path)

        self.img_root = img_root
        self.obj_root = obj_root
        self.dataset_size = len(self.img_list)

        self.transform = transforms.ToTensor()
        self.sample_nPoint = 3000

        self.locations = defaultdict(str)
        #self.__load_dataset()

    def __load_dataset(self):
        for idx in range(self.dataset_size):
            if idx % 10000 == 0:
                print('loaded', str(idx // self.dataset_size * 100)+'%')
            line = self.img_list[idx].strip()
            img_path = os.path.join(self.img_root, self.mode, 'rgb_obj', line)
            meta_path = img_path.replace('rgb_obj', 'meta').replace('jpg', 'pkl')
            meta = pickle.load(open(meta_path, 'rb'))

            obj_path = meta['obj_path']
            obj_path_seg = obj_path.split('/')[4:]
            obj_path_seg = [it + '/' for it in obj_path_seg]
            obj_mesh_path = ''.join(obj_path_seg)[:-1]
            obj_mesh_path = os.path.join(self.obj_root, obj_mesh_path)
            save_name = obj_mesh_path.replace('model_normalized', 'contactmap').replace('obj', 'npy')
            self.locations[str(idx)] = save_name

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        line = self.img_list[idx].strip()
        img_path = os.path.join(self.img_root, self.mode, 'rgb_obj', line)
        meta_path = img_path.replace('rgb_obj', 'meta').replace('jpg', 'pkl')
        #img = Image.open(img_path)
        meta = pickle.load(open(meta_path, 'rb'))

        # hand information
        # hand_pose = torch.tensor(meta['hand_pose'])
        # hand_shape = torch.tensor(meta['shape'])
        # hand_trans = torch.tensor(meta['trans'])
        # hand_info = torch.cat((hand_pose, hand_shape), dim=0) # [55]
        # hand_side = meta['side']
        #hand_xyz = torch.tensor(meta['verts_3d']).permute(1,0) # [778, 3] -> [3, 778]

        # object information
        obj_id = meta['sample_id']

        obj_path = meta['obj_path']
        obj_path_seg = obj_path.split('/')[4:]
        obj_path_seg = [it + '/' for it in obj_path_seg]
        obj_mesh_path = ''.join(obj_path_seg)[:-1]
        obj_mesh_path = os.path.join(self.obj_root, obj_mesh_path)
        obj_xyz_normalized = np.array(utils.fast_load_obj(open(obj_mesh_path))[0]['vertices']) # [N, 3]
        obj_xyz_normalized = torch.tensor(obj_xyz_normalized, dtype=torch.float32)
        nPoint = obj_xyz_normalized.size(0)

        obj_scale = meta['obj_scale']
        obj_scale_tensor = torch.tensor(obj_scale).type_as(obj_xyz_normalized).repeat(nPoint, 1)  # [N, 1]
        obj_pc = torch.cat((obj_xyz_normalized, obj_scale_tensor), dim=-1).permute(1, 0) # [4,N]

        #save_name = obj_mesh_path.replace('model_normalized', 'contactmap').replace('obj','npy')
        #prediction_exist = os.path.isfile(save_name)
        #self.locations[str(idx)] = self.locations.get()save_name
        # if prediction_exist:
        #     exist = 1
        # else:
        #     exist = 0

        return (obj_pc, torch.tensor(idx))

