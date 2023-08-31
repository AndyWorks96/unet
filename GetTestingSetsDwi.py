import os
import numpy as np
import SimpleITK as sitk

dwi_name = "_dwi.nii.gz"

mask_name = "_seg.nii.gz"



bratshgg_path = './data/dwiTest'

outputImg_path = './data/dwiImage01/'
outputMask_path = './data/dwiMask01/'

if not os.path.exists(outputImg_path):
    os.mkdir(outputImg_path)
if not os.path.exists(outputMask_path):
    os.mkdir(outputMask_path)


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


pathhgg_list = file_name_path(bratshgg_path)
# pathlgg_list = file_name_path(bratslgg_path)


def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    #有点像“去掉最低分去掉最高分”的意思,使得数据集更加“公平”
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)#限定范围numpy.clip(a, a_min, a_max, out=None)

    #除了黑色背景外的区域要进行标准化
    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9 #黑色背景区域
        return tmp


def crop_ceter(img,croph,cropw):
    #for n_slice in range(img.shape[0]):
    height,width = img[0].shape
    starth = height//2-(croph//2)
    startw = width//2-(cropw//2)
    return img[:,starth:starth+croph,startw:startw+cropw]


for subsetindex in range(len(pathhgg_list)):
    brats_subset_path = bratshgg_path + "/" + str(pathhgg_list[subsetindex]) + "/"
    # 获取每个病例的四个模态及Mask的路径
    dwi_image = brats_subset_path + str(pathhgg_list[subsetindex]) + dwi_name

    mask_image = brats_subset_path + str(pathhgg_list[subsetindex]) + mask_name
    # 获取每个病例的四个模态及Mask数据
    dwi_src = sitk.ReadImage(dwi_image, sitk.sitkInt16)

    mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)
    # GetArrayFromImage()可用于将SimpleITK对象转换为ndarray
    dwi_array = sitk.GetArrayFromImage(dwi_src)

    mask_array = sitk.GetArrayFromImage(mask)
    # 对四个模态分别进行标准化,由于它们对比度不同
    dwi_array_nor = normalize(dwi_array)

    # 裁剪(偶数才行)
    dwi_crop = crop_ceter(dwi_array_nor, 224, 224)

    mask_crop = crop_ceter(mask_array, 224, 224)
    print(str(pathhgg_list[subsetindex]))
    # 切片处理,并去掉没有病灶的切片
    for n_slice in range(dwi_crop.shape[0]):
        if np.max(mask_crop[n_slice, :, :]) != 0:
            maskImg = mask_crop[n_slice, :, :]

            OneModelImageArray = np.zeros((dwi_crop.shape[1], dwi_crop.shape[2], 1), np.float)
            dwiImg = dwi_crop[n_slice, :, :]
            dwiImg = dwiImg.astype(np.float)
            OneModelImageArray[:, :, 0] = dwiImg


            imagepath = outputImg_path + str(pathhgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
            maskpath = outputMask_path  + str(pathhgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
            np.save(imagepath, OneModelImageArray)  # (160,160,4) np.float dtype('float64')
            np.save(maskpath, maskImg)  # (160, 160) dtype('uint8') 值为0 1 2 4
print("Done！")