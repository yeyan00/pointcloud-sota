import os
import collections
import os,sys

# pointcept not install,just add it's directory to sys.path
pointcept_dir = ''
sys.path.append(pointcept_dir)

import torch
import pointcept.utils.comm as comm
from pointcept.engines.defaults import (
    default_config_parser,
    default_setup,
)
from pointcept.models import build_model
import numpy as np
import json



class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")
        
def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(
        arr.shape[0], dtype=np.uint64
    )
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr
    
def get_model(cfg):
    model = build_model(cfg.model).cuda()
    if os.path.isfile(cfg.weight):
        checkpoint = torch.load(cfg.weight)
        state_dict = checkpoint["state_dict"]
        new_state_dict = collections.OrderedDict()
        for name, value in state_dict.items():
            if name.startswith("module."):
                if comm.get_world_size() == 1:
                    name = name[7:]  # module.xxx.xxx -> xxx.xxx
            else:
                if comm.get_world_size() > 1:
                    name = "module." + name  # xxx.xxx -> module.xxx.xxx
            new_state_dict[name] = value
        model.load_state_dict(new_state_dict, strict=True)
    return model

def get_test_data(data_root):
    meta_json = '%s/%s.json'%(data_root,'val')
    data_list=[]
        
    with open(meta_json) as json_file:
        data = json.load(json_file)
        for k,v in data.items():
            for i in v:
                data_list.append(i)

    return data_list

def load_data(data):
    lidar_pths = data['data']
    xyz = []
    cls = []
    for i in lidar_pths:
        fp = np.load(i)
        _xyz = fp['xyz']
        _cls = fp['cls']
        xyz.append(_xyz)
        cls.append(_cls)

    coord = np.vstack(xyz).astype(np.float32) #[:, :3]
    strength =  np.ones((coord.shape[0],), dtype=np.int32).reshape(-1,1)
    segment = np.hstack(cls).reshape(-1).astype(np.int64)
    
    data_dict = dict(coord=coord, strength=strength,segment=segment)
    
    return data_dict
    
def pre_process_data(data,grid_size=0.4):
    data_dict = data
    scaled_coord = data_dict["coord"] / np.array(grid_size)
    discrete_coord = np.floor(scaled_coord).astype(int)
    min_coord = discrete_coord.min(0) * np.array(grid_size)
    discrete_coord -= discrete_coord.min(0)
    key = fnv_hash_vec(discrete_coord)
    _, idx_unique,inverse = np.unique(key, return_index=True,return_inverse=True)
    
    to = ToTensor()
 
    data_dict["discrete_coord"] = to(discrete_coord[idx_unique])

   
    for key in ("coord", "strength", "segment"):
        data_dict[key] = to(data_dict[key][idx_unique])
        
    
    
    data['offset'] = torch.tensor([data_dict["coord"].shape[0]])
     
    data_dict['feat'] = torch.cat([data_dict[key].float() for key in ("coord", "strength")], dim=1)
    #data_dict['inv'] = inverse
    return data_dict,inverse
         
    
if __name__=="__main__":
    
    work_dir = 'workspace/tsunet' 
    config_file = 'ts_unet_cfg.py'
    options = {
        'resume':True,
        'weight':f'{work_dir}/model/model_last.pth'
    }
    
    test_data = 'test'
    
    data_list = get_test_data(test_data)
    
    data = load_data(data_list[100])
    
    src_xyz = data['coord']#.copy()
    cls = data['segment']
    
    data_dict,inv = pre_process_data(data)
    
    for key in data_dict.keys():
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].cuda(non_blocking=True)
            
    cfg = default_config_parser(config_file,options)
    cfg = default_setup(cfg)
    
    model = get_model(cfg)

    model.eval()
    
    output = model(data_dict)
    pred = output['seg_logits'].max(1)[1].cpu().numpy()
    
    voxel_xyz = data_dict['coord'].cpu().numpy()
    voxel_cls = data_dict['segment'].cpu().numpy()
    
    # convert to orgin pointcloud
    pred_cls = -np.ones(len(src_xyz), dtype='i4') 
    pred_cls[...]=pred[inv]
    


