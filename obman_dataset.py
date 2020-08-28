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
            #self.file_path = '/hand-object-3/hanwen/contactdb_prediction/logs/2020_8_24_9_train/log_unprocessed.txt'
            self.file_path = '/hand-object-3/download/dataset/ObMan/obman/train.txt'
        elif self.mode == "val":
            #self.file_path = '/hand-object-3/hanwen/contactdb_prediction/logs/2020_8_24_7_val/log_unprocessed.txt'
            self.file_path = '/hand-object-3/download/dataset/ObMan/obman/val.txt'
        else:
            #self.file_path = '/hand-object-3/hanwen/contactdb_prediction/logs/2020_8_24_8_test/log_unprocessed.txt'
            self.file_path = '/hand-object-3/download/dataset/ObMan/obman/test.txt'
        self.img_list = utils.readTxt_obman(self.file_path)

        self.img_root = img_root
        self.obj_root = obj_root
        self.dataset_size = len(self.img_list)

        self.transform = transforms.ToTensor()
        self.sample_nPoint = 3000

        self.locations = defaultdict(str)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        line = self.img_list[idx].strip()
        img_path = os.path.join(self.img_root, self.mode, 'rgb_obj', line)
        meta_path = img_path.replace('rgb_obj', 'meta').replace('jpg', 'pkl')
        #img = Image.open(img_path)
        meta = pickle.load(open(meta_path, 'rb'))

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

        return (obj_pc, torch.tensor(idx))

