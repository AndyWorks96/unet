import pydicom
import matplotlib.pyplot as plt
import scipy.misc
import pandas as pd
import numpy as np
import os
import imageio
import SimpleITK as sitk
import tqdm
from PIL import Image
import cv2

file_path01 = './data/dcmT2'
file_path02='./data/t2mask'
# output_pngpath = './data/png'

def mkDir(outputpath):
    if not os.path.exists(outputpath):
        os.mkdir(outputpath)
        # return outputpath


def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):# 目录遍历器
        if len(dirs) and dir:# 意思目录不为空才输出
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files

path_list01 = file_name_path(file_path01)
path_list02 = file_name_path(file_path02)

def getPatientDir():

    for subsetindex in range(len(path_list01)):
        retal_subset_path01 = file_path01 + "/" + str(path_list01[subsetindex]) + "/"+"T2/"
        retal_subset_path02 = file_path02 + "/" + str(path_list01[subsetindex])+"/"
        # 创建患者png路径
        # outPatientone = mkDir(output_pngpath+"/"+str(path_list[subsetindex]))
        # return retal_subset_path

        c = []
        namesMask = os.listdir(retal_subset_path02)  # 路径
        patientLen = len(namesMask)
        OnePeople = np.zeros([patientLen, 512, 512], dtype=np.uint8)
        OneMask = np.zeros([patientLen, 512, 512], dtype=np.uint8)

        # 将文件夹中的文件名称与后边的 .dcm分开
        for name in namesMask:
            if name == '.DS_Store':
                continue
            index = name.rfind('.')
            name01 = name[:index]

            c.append(name01)

        for files in c:

            picture_path = retal_subset_path01 + files + ".dcm"
            dwi = sitk.ReadImage(picture_path, sitk.sitkInt16)
            # out_path = output_pngpath+"/"+str(path_list[subsetindex]) +"/"+ files + ".png"
            dwiMask_path = retal_subset_path02 + files +".png"
            I1 = Image.open(dwiMask_path)
            # I1.show()
            I_array = np.array(I1)
            I_array = I_array + 0
            # I_array.astype(int)
            indexFiles = c.index(files)

            print(indexFiles)
            dwi_array = sitk.GetArrayFromImage(dwi)
            x = dwi_array.shape[1]
            ddd = dwi_array[0]
            # for idz in range(OneMask.shape[0]):
            #
            for idx in range(OneMask.shape[1]):
                for idy in range(OneMask.shape[2]):
                    OnePeople[indexFiles,idx,idy] = dwi_array[0][idx][idy]
                    OneMask[indexFiles, idx, idy] = I_array[idx][idy]
        saveOnePeople = sitk.GetImageFromArray(OnePeople)
        saveOneMask = sitk.GetImageFromArray(OneMask)
        savedir = './data/t2Test/'+ str(path_list01[subsetindex]) + '/'
        sitk.WriteImage(saveOnePeople, savedir + str(path_list01[subsetindex]) + "_t2.nii.gz")
        sitk.WriteImage(saveOneMask, savedir + str(path_list01[subsetindex]) + "_seg.nii.gz")
        print('all is changed')




#
path = getPatientDir()
# Dcm2jpg(path)