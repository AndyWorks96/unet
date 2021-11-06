
import six
import pandas as pd
from radiomics import featureextractor
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

file_path = './data/t2Test'

def output_xlsx(dict_item,name):
    pf=pd.DataFrame(list(dict_item))
    xlsx_name=name+'.xlsx'
    xlsx_obj=pd.ExcelWriter(xlsx_name)
    pf.to_excel(xlsx_obj)
    xlsx_obj.save()


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

def getPaRadio():
    path_list = file_name_path(file_path)
    params = './data/pyradiomics_yaml/exampleMR_NoResampling.yaml'

    for subsetindex in range(len(path_list)):

        retal_subset_path = file_path + "/" + str(path_list[subsetindex])+"/"
        imageName = retal_subset_path +str(path_list[subsetindex])+"_t2.nii.gz"
        maskName = retal_subset_path +str(path_list[subsetindex])+"_seg.nii.gz"
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
        result = extractor.execute(imageName, maskName)
        dict_item = result.items()
        name = retal_subset_path + str(path_list[subsetindex])
        output_xlsx(dict_item, name)
        print('all is changed')


getPaRadio()

# for key, val in six.iteritems(result):
#     print("\t%s: %s" %(key, val))
