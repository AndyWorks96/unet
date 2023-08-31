
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

# def output_xlsx(dict_item,name):
#     dict = list(dict_item)
#     data_frame = None
#     if data_frame is None:
#
#         data_frame = pd.DataFrame(dict)
#
#     # pf=pd.DataFrame(list(dict_item))
#     # data_frame['label'] = 0
#
#     xlsx_name=name+'.xlsx'
#     xlsx_obj=pd.ExcelWriter(xlsx_name)
#     data_frame = data_frame[22:122]
#     # data_frame.loc['123'] = ['label', 1]
#     # data_frameT = data_frame.T
#     # data_frameT = data_frameT.loc[0]
#     # pf.to_excel(xlsx_obj)
#     data_frameT.to_excel(xlsx_obj, header=None, index=False)
#     xlsx_obj.save()



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

    # all_data_frame = pd.DataFrame( np.arange(200).reshape(100,2))
    # all_data_frame = all_data_frame[22:122]
    imageName01 = './data/t2Test/REN GUI LIAN^REN GUI/REN GUI LIAN^REN GUI_t2.nii.gz'
    maskName01 = './data/t2Test/REN GUI LIAN^REN GUI/REN GUI LIAN^REN GUI_seg.nii.gz'
    extractor01 = featureextractor.RadiomicsFeatureExtractor(params)
    result01 = extractor01.execute(imageName01, maskName01)

    dict_item01 = result01.items()
    all_data_frame = pd.DataFrame(list(dict_item01))[22:]
    all_data_frame.loc['724'] = ['name', 'REN GUI LIAN^REN GUI' ]

    for subsetindex  in range(len(path_list)):
        if subsetindex == 0:
            continue
        retal_subset_path = file_path + "/" + str(path_list[subsetindex])+"/"
        imageName = retal_subset_path +str(path_list[subsetindex])+"_t2.nii.gz"
        maskName = retal_subset_path +str(path_list[subsetindex])+"_seg.nii.gz"
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
        result = extractor.execute(imageName, maskName)
        dict_item = result.items()


        dict = list(dict_item)
        data_frame = None
        if data_frame is None:
            data_frame = pd.DataFrame(dict)

        # pf=pd.DataFrame(list(dict_item))
        # data_frame['label'] = 0

        data_frame = data_frame[22:]

        data_frame.loc['724'] = ['name',path_list[subsetindex]]
        # data_frame[len(data_frame.T)] = data_frame.loc[0]
        all_data_frame[len(all_data_frame.T)] = data_frame.T.loc[1]
        # data_frame.loc['123'] = ['label', 1]
        # data_frameT = data_frame.T
        # data_frameT = data_frameT.loc[0]
        # radio_name = all_data_frame[0]
        # radio_22_name = radio_name[22]

        if subsetindex == 42:
            name = './data/t2Test/' + 'allPatientsRadio_t2'

            xlsx_name = name  + '.xlsx'
            xlsx_obj = pd.ExcelWriter(xlsx_name)

            current = 22
            for idx in range(current, 724):
                all_data_frame[0][idx] = 'T2_' + all_data_frame[0][idx]
            all_data_frame = all_data_frame.T
            all_data_frame.to_excel(xlsx_obj, header=None, index=False)

            xlsx_obj.save()


        # print('all is changed')


getPaRadio()
print('123')
print('456')
# for key, val in six.iteritems(result):
#     print("\t%s: %s" %(key, val))
