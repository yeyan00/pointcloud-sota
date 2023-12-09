"""
Main Training Script
"""
import os,sys

# pointcept not install,just add it's directory to sys.path
pointcept_dir = ''
sys.path.append(pointcept_dir)

from dataset import MyDataset

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import Trainer
from pointcept.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = Trainer(cfg)
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    
    # set work directory
    work_dir = 'workspace' 
    
    # set gpu VISIBLE DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"  
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # print error info
    
    args.num_gpus = 4
    
    # torchsparse unet
    args.config_file = 'ts_unet_cfg.py' 
    

    args.options = {
        'save_path':f'{work_dir}/log/tsunet', 
        #'resume':True,  # 
        #'weight':f'{work_dir}/model/model_last.pth'
  
    }
    
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
