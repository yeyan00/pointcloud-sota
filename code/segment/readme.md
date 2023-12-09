for pointcloud segment, i use pointcept
* [pointcept](https://github.com/Pointcept/Pointcept) 


**The following is how I use pointcept library,add my dataset and model(torchsparse unet)**

# 1.install
refer to pointcept doc:  
https://github.com/Pointcept/Pointcept?tab=readme-ov-file#installation

# 2.define your dataset
**refer to dataset.py**

# 3.define your model [option]
**refer to ts_unet.py**

i use torchspaser , pointcept default is spconv(**now recommend**) and mink. 

spconv V2.3.6 speed faster than torchsparse V2.0, but slower than torchsparse V2.1,but torchsparse V2.1 sometimes make mistakes in traininng

* [torchsparse](https://github.com/mit-han-lab/torchsparse)  


# 4.define config file
**refer to ts_unet_cfg.py**

```
batch_size = 24
epoch = 200
eval_epoch = 200
dataset_type = "MyDataset"
data_root = "dataset/"
ignore_index = -1
names = [
    'bg',
    "c1",
    "c2"
]

```

# 5.train
**refer to train.py**
```
set your pointcept_dir,args.config_file,work_dir,args.num_gpus

```

# 6.infer
**refer to infer.py**
```
set your pointcept_dir,args.config_file,work_dir
```





