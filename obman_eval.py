from models.voxnet import DiverseVoxNet as VoxNet
from models.pointnet import DiversePointNet as PointNet
from voxel_dataset import VoxelDataset
from pointcloud_dataset import PointCloudDataset
from obman_dataset import obman
from models.losses import DiverseLoss
import obman_utils
import numpy as np
import open3d
import os
import torch
from torch.utils.data import DataLoader
import argparse
import configparser
import pickle
from IPython.core.debugger import set_trace
import time
osp = os.path

torch.backends.cudnn.enabled = False

def show_pointcloud_texture(geom, tex_preds):
  cmap = np.asarray([[0, 0, 1], [1, 0, 0]]) # rgb
  x, y, z, scale = geom
  pts = np.vstack((x, y, z)).T * scale[0]
  for tex_pred in tex_preds:
    pc = open3d.PointCloud()
    pc.points = open3d.Vector3dVector(pts)
    tex_pred = np.argmax(tex_pred, axis=0)
    tex_pred = cmap[tex_pred]
    pc.colors = open3d.Vector3dVector(tex_pred)
    open3d.draw_geometries([pc])


def show_voxel_texture(geom, tex_preds):
  cmap = np.asarray([[0, 0, 1], [1, 0, 0]])
  z, y, x = np.nonzero(geom[0])
  pts = np.vstack((x, y, z)).T
  for tex_pred in tex_preds:
    tex_pred = np.argmax(tex_pred, axis=0)
    tex_pred = tex_pred[z, y, x]
    tex_pred = cmap[tex_pred]
    pc = open3d.PointCloud()
    pc.points = open3d.Vector3dVector(pts)
    pc.colors = open3d.Vector3dVector(tex_pred)
    open3d.draw_geometries([pc])


def eval_obman(data_dir, instruction, checkpoint_filename, config_filename, device_id,
    test_only=False, mode='train'):
  # config
  config = configparser.ConfigParser()
  config.read(config_filename)
  droprate  = config['hyperparams'].getfloat('droprate')

  # cuda
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  print('using device', device)

  # load checkpoint
  checkpoint = torch.load(checkpoint_filename, map_location=torch.device('cpu'))

  # create model
  model_name = osp.split(config_filename)[1].split('.')[0]
  kwargs = dict(data_dir=data_dir, instruction=instruction, train=False,
    random_rotation=0, n_ensemble=-1, test_only=test_only)
  if 'voxnet' in model_name:
    model = VoxNet(n_ensemble=checkpoint.n_ensemble, droprate=droprate)
    model.voxnet.load_state_dict(checkpoint.voxnet.state_dict())
    grid_size = config['hyperparams'].getint('grid_size')
    dset = VoxelDataset(grid_size=grid_size, **kwargs)
  elif 'pointnet' in model_name:
    model = PointNet(n_ensemble=checkpoint.n_ensemble, droprate=droprate)
    model.pointnet.load_state_dict(checkpoint.pointnet.state_dict())
    n_points = config['hyperparams'].getint('n_points')
    #dset = PointCloudDataset(n_points=n_points, random_scale=0, **kwargs)
    dset = obman(mode=mode)
  else:
    raise NotImplementedError
  if 'pointnet' not in model_name:
    model.eval()
  model.to(device=device)
  model.eval()
  dloader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=8)

  local_time = time.localtime(time.time())
  time_str = str(local_time[0]) + '_' + str(local_time[1]) + '_' + str(local_time[2]) + '_' + str(local_time[3])
  save_root = './logs/' + time_str + '_' + mode
  if not os.path.exists(save_root):
    os.makedirs(save_root)
  log_root = save_root + '/log_unprocessed.txt'
  log_file = open(log_root, 'w+')
  log_file.close()

  unprocessed = set()

  img_list = obman_utils.get_img_list_val(mode)
  for batch_idx, (obj_pc, idx) in enumerate(dloader):
    B, D, N = obj_pc.size() # set B=1
    if N >= 100000:
      if idx.numpy()[0] not in unprocessed:
        unprocessed.add(idx.nunmpy()[0])
        out_str = str(idx.numpy()[0])
        with open(log_root, 'a') as f:
          f.write(out_str + '\n')
        print('idx {} exceed size'.format(out_str))
      continue
    if B != 1:
      print('wrong batch size', B)
    line = img_list[idx.numpy()[0]]
    save_name = obman_utils.get_saveName(line, mode)
    #save_name = dset.locations[str(idx.numpy()[0])]
    if os.path.isfile(save_name):
      continue # already predicted on this object model
    else:
      with torch.no_grad():
        obj_pc = obj_pc.to(device)
        tex_preds = model(obj_pc) # [1,10,2,N]
        save_tensor = tex_preds.cpu().numpy().squeeze() # [10,2,N]
        save_tensor = np.argmax(save_tensor, axis=1) # [10,1,N], dim2: 1 for positive
        if batch_idx % 1000 == 0:
          print(str(batch_idx/len(dloader)*100) + '%', save_name)
        #np.save(save_tensor, save_name)
        np.save(save_name, save_tensor)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', default=osp.join('data', 'voxelized_meshes'))
  parser.add_argument('--instruction', type=str, default='use')
  parser.add_argument('--checkpoint_filename', type=str, default='data/checkpoints/use_pointnet_diversenet/model_pointnet.pth')
  parser.add_argument('--config_filename', type=str, default='configs/pointnet.ini')
  parser.add_argument('--test_only', action='store_true')
  parser.add_argument('--device_id', default=0)
  parser.add_argument('--show_object', default=None)
  parser.add_argument('--mode', type=str, default='train')
  args = parser.parse_args()

  eval_obman(osp.expanduser(args.data_dir), args.instruction,
             osp.expanduser(args.checkpoint_filename),
             osp.expanduser(args.config_filename), args.device_id,
             test_only=args.test_only, mode=args.mode)
