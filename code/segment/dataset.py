# segment use pointcept
# https://github.com/Pointcept/Pointcept


# 1.define dataset
import os
import numpy as np
from collections.abc import Sequence
import pickle
from pointcept.datasets.builder import DATASETS
from pointcept.datasets.defaults import DefaultDataset
import json

'''
dataset like this:

train.txt
data/1.npz
data/2.npz

in npz,data key: xyz,c
'''

@DATASETS.register_module()
class MyDataset(DefaultDataset):
    def __init__(
        self,
        split="train",
        data_root="data/dataset",
        sweeps=10,
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
    ):
        self.sweeps = sweeps
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )


    def get_data_list(self):
        meta_json = '%s/%s/%s.txt'%(self.data_root,self.split,self.split)
        
        with open(meta_json,'r') as f:
           data_list=  f.readlines()
        
        data_list = [e.strip() for e in data_list]
        
        return data_list

    def get_data(self, idx):
        fname = self.data_list[idx % len(self.data_list)]        
        with  np.load(fname) as buf:
            coord = buf['xyz'].astype('f4')  # Nx3
            _c = buf['cls']
            
            segment = np.zeros(_c.shape)      
            segment = segment.astype(np.int64)
    
        # just use as one
        strength =  np.ones((coord.shape[0],), dtype=np.int32).reshape(-1,1)
        segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(
                np.int64
            )
        
        data_dict = dict(coord=coord, strength=strength,segment=segment)
        return data_dict

    def get_data_name(self, idx):
        # return data name for lidar seg, optimize the code when need to support detection
        _name = os.path.basename(self.data_list[idx % len(self.data_list)])
        return _name

    @staticmethod
    def get_learning_map(ignore_index):
        # remap cls
        learning_map = {
            -2: ignore_index,
            -1: ignore_index,
            0: 0,
            1: 1,
            2: 2 
        }
        return learning_map

    