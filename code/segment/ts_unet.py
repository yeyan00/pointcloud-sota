import numpy as np
import torch
import torch.autograd.profiler as profiler
import torch.cuda
import torch.nn as nn
import torch.optim
from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch

from timm.models.layers import trunc_normal_

import torchsparse
import torchsparse.nn as spnn
from torchsparse import SparseTensor
#from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.backbones import SparseResUNet42

ts_version = torchsparse.__version__
if ts_version>='2.1.0':
    import torchsparse.nn.functional as F
    F.set_kmap_mode('hashmap')
    F.set_conv_mode(2)
    F.set_downsample_mode('spconv')	
    

# kmap_mode = F.get_kmap_mode()
# print(kmap_mode)

@MODELS.register_module("TSUNet-V0")
class SparseUnetSeg(nn.Module):
    def __init__(self,in_channels=4,num_classes=2,**kwargs) -> None:
        super().__init__()
        self.backbone = SparseResUNet42(in_channels=in_channels,width_multiplier=1.0)
        self.header = spnn.Conv3d(self.backbone.decoder_channels[-1], num_classes,kernel_size=1,bias=True)
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, spnn.BatchNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, data_dict):
        discrete_coord = data_dict["discrete_coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"]
        batch = offset2batch(offset)
        
        x = SparseTensor(coords=discrete_coord.int(), feats=feat)
        
        # version>2.1.0 batch index is 0
        if ts_version>='2.1.0':
            x.coords =  torch.cat((batch.reshape(-1,1),x.coords), dim=1).int()
        else:
            # version<=2.0.0
            x.coords =  torch.cat((x.coords, batch.reshape(-1,1)), dim=1).int()
        back_out = self.backbone(x)
        
        output = self.header(back_out[-1])
        
        return output.F