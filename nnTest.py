import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset
import unet_5k
import unet
from metrics import dice_coef, batch_iou, mean_iou, iou_score ,ppv,sensitivity
import losses
from utils import str2bool, count_params
# from sklearn.externals import joblib
#from hausdorff import hausdorff_distance
import imageio
#import ttach as tta
import SimpleITK as sitk
wt_dices = []
tc_dices = []
et_dices = []
wt_sensitivities = []
tc_sensitivities = []
et_sensitivities = []
wt_ppvs = []
tc_ppvs = []
et_ppvs = []
wt_Hausdorf = []
tc_Hausdorf = []
et_Hausdorf = []

def hausdorff_distance(lT,lP):
    labelPred=sitk.GetImageFromArray(lP, isVector=False)
    labelTrue=sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    return hausdorffcomputer.GetAverageHausdorffDistance()
    # hausdorffcomputer.GetHausdorffDistance()


def CalculateWTTCET(wtpbregion,wtmaskregion,tcpbregion,tcmaskregion,etpbregion,etmaskregion):
    # 开始计算WT
    dice = dice_coef(wtpbregion,wtmaskregion)
    wt_dices.append(dice)
    ppv_n = ppv(wtpbregion, wtmaskregion)
    wt_ppvs.append(ppv_n)
    # Hausdorff = hausdorff_distance(wtmaskregion, wtpbregion)
    # wt_Hausdorf.append(Hausdorff)
    sensitivity_n = sensitivity(wtpbregion, wtmaskregion)
    wt_sensitivities.append(sensitivity_n)
    # 开始计算TC
    dice = dice_coef(tcpbregion, tcmaskregion)
    tc_dices.append(dice)
    ppv_n = ppv(tcpbregion, tcmaskregion)
    tc_ppvs.append(ppv_n)
    # Hausdorff = hausdorff_distance(tcmaskregion, tcpbregion)
    # tc_Hausdorf.append(Hausdorff)
    sensitivity_n = sensitivity(tcpbregion, tcmaskregion)
    tc_sensitivities.append(sensitivity_n)
    # 开始计算ET
    dice = dice_coef(etpbregion, etmaskregion)
    et_dices.append(dice)
    ppv_n = ppv(etpbregion, etmaskregion)
    et_ppvs.append(ppv_n)
    # Hausdorff = hausdorff_distance(etmaskregion, etpbregion)
    # et_Hausdorf.append(Hausdorff)
    sensitivity_n = sensitivity(etpbregion, etmaskregion)
    et_sensitivities.append(sensitivity_n)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')
    parser.add_argument('--mode', default=None,
                        help='GetPicture or Calculate')

    args = parser.parse_args()

    return args

def main():
    val_args = parse_args()
    val_args.name = 'AndyWorks_Unet_woDS'
    # val_args.mode = "GetPicture"
    val_args.mode = "Calculate"
    args = joblib.load('models/%s/args.pkl' %val_args.name)

    if not os.path.exists('output/%s' %args.name):
        os.makedirs('output/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    if val_args.mode == "Calculate":
        """
        计算各种指标:Dice、Sensitivity、PPV
        """

        maskPath = glob('./output/label/*.nii.gz')
        pbPath = glob('output/predict/*.nii.gz')

        if len(maskPath) == 0:
            print("请先生成图片!")
            return
        masklength=len(maskPath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            with torch.no_grad():
                for myi in tqdm(range(len(maskPath))):

                    mask = sitk.ReadImage(maskPath[myi])
                    pb = sitk.ReadImage(pbPath[myi])
                    pix_mask = sitk.GetArrayFromImage(mask)
                    pix_pb = sitk.GetArrayFromImage(pb)

                    # 创建三个全黑的三维矩阵，分别用于预测出来的WT、TC、ET分块的拼接
                    OneWT = np.zeros([155, 240, 240], dtype=np.uint8)
                    OneTC = np.zeros([155, 240, 240], dtype=np.uint8)
                    OneET = np.zeros([155, 240, 240], dtype=np.uint8)
                    # 创建三个全黑的三维矩阵，分别用于真实的WT、TC、ET分块的拼接
                    OneWTMask = np.zeros([155, 240, 240], dtype=np.uint8)
                    OneTCMask = np.zeros([155, 240, 240], dtype=np.uint8)
                    OneETMask = np.zeros([155, 240, 240], dtype=np.uint8)
                    for idx in range(pix_pb.shape[0]):
                        for idy in range(pix_pb.shape[1]):
                            for idz in range(pix_pb.shape[2]):

                                # 只要这个像素的任何一个通道有值,就代表这个像素不属于前景,即属于WT区域
                                # if pix_mask[idx, idy, idz:].any() != 0:
                                #     OneWTMask[idx, idy,idz] = 1
                                # if pix_pb[idx, idy, idz:].any() != 0:
                                #     OneWT[idx, idy,idz] = 1
                                # # 判断肿瘤核心
                                # if pix_mask[idx, idy, idz] == 1 or pix_mask[idx, idy, idz]==4:
                                #     OneTCMask[idx, idy,idz] = 1
                                # if pix_pb[idx, idy, idz] ==1 or pix_pb[idx, idy, idz]==4:
                                #     OneTC[idx, idy,idz] = 1
                                # ET区域
                                if pix_mask[idx, idy, idz] == 4:
                                    OneETMask[idx, idy,idz] = 1
                                if pix_pb[idx, idy, idz] == 4:
                                    OneET[idx, idy,idz] = 1
                    #开始计算
                    CalculateWTTCET(OneWT, OneWTMask, OneTC, OneTCMask, OneET, OneETMask)
                torch.cuda.empty_cache()

        print('WT Dice: %.4f' % np.mean(wt_dices))
        print('TC Dice: %.4f' % np.mean(tc_dices))
        print('ET Dice: %.4f' % np.mean(et_dices))
        print("=============")
        print('WT PPV: %.4f' % np.mean(wt_ppvs))
        print('TC PPV: %.4f' % np.mean(tc_ppvs))
        print('ET PPV: %.4f' % np.mean(et_ppvs))
        print("=============")
        print('WT sensitivity: %.4f' % np.mean(wt_sensitivities))
        print('TC sensitivity: %.4f' % np.mean(tc_sensitivities))
        print('ET sensitivity: %.4f' % np.mean(et_sensitivities))
        print("=============")
        # print('WT Hausdorff: %.4f' % np.mean(wt_Hausdorf))
        # print('TC Hausdorff: %.4f' % np.mean(tc_Hausdorf))
        # print('ET Hausdorff: %.4f' % np.mean(et_Hausdorf))
        print("=============")


if __name__ == '__main__':
    main( )
