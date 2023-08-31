import numpy as np
import cv2 #https://www.jianshu.com/p/f2e88197e81d
import random

from skimage.io import imread
from skimage import color

import torch
import torch.utils.data

import torch.nn as nn
from torchvision import datasets, models, transforms







class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        # self.conv1 = nn.Conv2d(4, 4, 3, padding=1)

    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        #读numpy数据(npy)的代码
        npimage = np.load(img_path)
        npmask = np.load(mask_path)
        npimage = npimage.transpose((2, 0, 1))
        # npimage = torch.from_numpy(npimage)
        # npimage = self.conv1(npimage)

        # ED2 ET4 NET1
        # WT = ED + ET + NET
        # TC = ET + NET
        # ET
        WT_Label = npmask.copy()
        WT_Label[npmask == 1] = 1.


        nplabel = np.empty((512, 512, 1))
        nplabel[:, :, 0] = WT_Label

        nplabel = nplabel.transpose((2, 0, 1))
        # np(3,160,160)
        # npimage = npimage.numpy()
        nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")

        return npimage,nplabel



