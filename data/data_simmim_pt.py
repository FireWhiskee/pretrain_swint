# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import math
import random
import numpy as np

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask
# --------------------------------------------------------
# SimMIM (Custom Dataset Edition)
# Modified for single-image training and custom datasets
# --------------------------------------------------------

import os
import math
import random
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

class CustomImageDataset(Dataset):
    """支持单图/自定义路径的增强型数据集"""
    def __init__(self, img_paths, img_size=224, repeat_factor=1000):
        """
        Args:
            img_paths (str/list): 单图路径或多图路径列表
            img_size (int): 模型输入尺寸
            repeat_factor (int): 单图重复次数模拟大数据集
        """
        if isinstance(img_paths, str):
            self.img_paths = [img_paths]
        else:
            self.img_paths = img_paths
            
        self.images = [Image.open(p).convert('RGB') for p in self.img_paths]
        self.img_size = img_size
        self.repeat_factor = repeat_factor
        
        # 动态增强策略（保留与ImageNet相同的归一化参数）
        self.base_transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.67, 1.), ratio=(3./4., 4./3.)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths) * self.repeat_factor

    def __getitem__(self, idx):
        # 循环使用图像
        img = self.images[idx % len(self.img_paths)].copy()  
        return self.base_transform(img)


# 修改CustomImageDataset类为支持文件夹的结构

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=224):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                         if f.endswith(('jpg', 'png'))]
        self.transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.67, 1.), ratio=(3./4., 4./3.)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        return self.transform(img)

class SimMIMTransform:
    """适配自定义数据集的增强转换"""
    def __init__(self, config):
        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=config.MODEL.SWIN.PATCH_SIZE,
            mask_ratio=config.DATA.MASK_RATIO
        )

    def __call__(self, img_tensor):
        # 在张量上生成掩码（与图像增强解耦）
        mask = self.mask_generator()
        return img_tensor, torch.from_numpy(mask).float()


# 修改build_loader_simmim函数
def build_loader_simmim(config):
    # 初始化数据集（使用整个训练文件夹）
    dataset = CustomImageDataset(
        img_dir="/content/drive/MyDrive/dior/data/train/class1",  # 指向训练集根目录
        img_size=config.DATA.IMG_SIZE
    )
    
    # 添加MIM转换层（保持原有逻辑）
    mim_transform = SimMIMTransform(config)
    dataset = TransformWrapper(dataset, mim_transform)
    
    # 分布式处理适配（保持原有逻辑）
    sampler = DistributedSampler(dataset) if dist.is_initialized() else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.DATA.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return dataloader

class TransformWrapper(Dataset):
    """转换流水线封装器"""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        return self.transform(img)

def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret
