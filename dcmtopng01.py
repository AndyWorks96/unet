import pydicom
import matplotlib.pyplot as plt
import scipy.misc
import pandas as pd
import numpy as np
import os
import imageio


file_path = './data/dcm'
output_pngpath = './data/t2Png'

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

path_list = file_name_path(file_path)


def getPatientDir():

    for subsetindex in range(len(path_list)):
        retal_subset_path = file_path + "/" + str(path_list[subsetindex]) + "/"+"T2/"

        # 创建患者png路径
        outPatientone = mkDir(output_pngpath+"/"+str(path_list[subsetindex]))
        # return retal_subset_path

        c = []
        names = os.listdir(retal_subset_path)  # 路径
        # 将文件夹中的文件名称与后边的 .dcm分开
        for name in names:
            index = name.rfind('.')
            name01 = name[:index]
            if name01 != "metacache" and name01 != "mimfancycache" and name01!=" ":
                c.append(name01)

        for files in c:
            picture_path = retal_subset_path + files + ".dcm"
            out_path = output_pngpath+"/"+str(path_list[subsetindex]) +"/"+ files + ".png"
            ds = pydicom.read_file(picture_path)
            img = ds.pixel_array  # 提取图像信息
            # scipy.misc.imsave(out_path, img)
            imageio.imwrite(out_path, img)
        print('all is changed')





path = getPatientDir()
Dcm2jpg(path)