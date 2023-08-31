import numpy as np
# from hausdorff import hausdorff_distance
import SimpleITK as sitk
import os
import glob
from medpy.metric.binary import hd95
import pandas as pd
from pandas import DataFrame


def dice(output, target):
    smooth = 1e-5
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def sensitivity(output, target):
    smooth = 1e-5
    intersection = (output * target).sum()
    return (intersection + smooth) / \
        (target.sum() + smooth)


def metric_hd(output, target):
    try:
        Hausdorff1 = hd95(output, target)
    except RuntimeError as e:
        print('RuntimeError, return 100')
        return 100
    return Hausdorff1


def hausdorff(output, target):
    labelPred = sitk.GetImageFromArray(output)
    labelTrue = sitk.GetImageFromArray(target)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    Hausdorff = hausdorffcomputer.GetHausdorffDistance()
    print(Hausdorff)
    return Hausdorff


def save_csv(file_path, dice_list, sensitivities_list, Hausdorf95_list, save_path):
    for i in range(len(file_path)):
        file_path[i] = file_path[i].split('\\')[-1]
    imgid_arr = np.array(file_path)[:, np.newaxis]
    dice_arr = np.array(dice_list)[:, np.newaxis]
    sensitivities_arr = np.array(sensitivities_list)[:, np.newaxis]
    Hausdorf95_arr = np.array(Hausdorf95_list)[:, np.newaxis]
    concatenate_array = np.concatenate((imgid_arr, dice_arr, sensitivities_arr, Hausdorf95_arr), axis=1)
    data = DataFrame(concatenate_array, columns=["image_id", "dice", "sensitivity", "hausdorf95"])
    data.to_csv(save_path)


def main():
    wt_dices = []
    tc_dices = []
    et_dices = []
    wt_sensitivities = []
    tc_sensitivities = []
    et_sensitivities = []
    wt_Hausdorf = []
    tc_Hausdorf = []
    et_Hausdorf = []

    outputs_path = glob.glob(r'seg_nii\*nii.gz')
    targets_path = glob.glob('./output/jiu0Monkey_Unet_woDS/*nii.gz')
    save_path = r'wt_index.csv'

    for i, j in zip(outputs_path, targets_path):
        output = sitk.GetArrayFromImage(sitk.ReadImage(i))
        target = sitk.GetArrayFromImage(sitk.ReadImage(j))

        wt_output_region = np.zeros([target.shape[0], target.shape[1], target.shape[2]], dtype=np.float32)
        wt_target_region = np.zeros([target.shape[0], target.shape[1], target.shape[2]], dtype=np.float32)

        tc_output_region = np.zeros([target.shape[0], target.shape[1], target.shape[2]], dtype=np.float32)
        tc_target_region = np.zeros([target.shape[0], target.shape[1], target.shape[2]], dtype=np.float32)

        et_output_region = np.zeros([target.shape[0], target.shape[1], target.shape[2]], dtype=np.float32)
        et_target_region = np.zeros([target.shape[0], target.shape[1], target.shape[2]], dtype=np.float32)

        wt_output_region = wt_output_region + output
        wt_output_region[wt_output_region > 0] = 1
        wt_target_region = wt_target_region + target
        wt_target_region[wt_target_region > 0] = 1

        tc_output_region = tc_output_region + output
        tc_output_region[tc_output_region == 2] = 0
        tc_output_region[tc_output_region > 0] = 1
        tc_target_region = tc_target_region + target
        tc_target_region[tc_target_region == 2] = 0
        tc_target_region[tc_target_region > 0] = 1

        et_output_region = et_output_region + output
        et_output_region[et_output_region == 1] = 0
        et_output_region[et_output_region == 2] = 0
        et_output_region[et_output_region > 0] = 1
        et_target_region = et_target_region + target
        et_target_region[et_target_region == 1] = 0
        et_target_region[et_target_region == 2] = 0
        et_target_region[et_target_region > 0] = 1

        print(i)
        dice_ = dice(wt_output_region, wt_target_region)
        wt_dices.append(dice_)
        sensitivity_n = sensitivity(wt_output_region, wt_target_region)
        wt_sensitivities.append(sensitivity_n)
        # print('wt:', dice_, sensitivity_n)
        Hausdorff = metric_hd(wt_output_region, wt_target_region)
        wt_Hausdorf.append(Hausdorff)
        print('wt:', dice_, sensitivity_n, Hausdorff)

        # dice_ = dice(tc_output_region, tc_target_region)
        # tc_dices.append(dice_)
        # sensitivity_n = sensitivity(tc_output_region, tc_target_region)
        # tc_sensitivities.append(sensitivity_n)
        # print('tc:', dice_, sensitivity_n)
        # Hausdorff = metric_hd(tc_output_region, tc_target_region)
        # tc_Hausdorf.append(Hausdorff)

        # dice_ = dice(et_output_region, et_target_region)
        # et_dices.append(dice_)
        # sensitivity_n = sensitivity(et_output_region, et_target_region)
        # et_sensitivities.append(sensitivity_n)
        # print('et:', dice_, sensitivity_n)
        # Hausdorff = metric_hd(et_output_region, et_target_region)
        # et_Hausdorf.append(Hausdorff)

    # outputs_path.append("mean_std")
    # wt_dices.append(np.mean(wt_dices))
    # wt_sensitivities.append(np.mean(wt_sensitivities))
    # wt_Hausdorf.append(np.mean(wt_Hausdorf))
    # save_csv(outputs_path, wt_dices, wt_sensitivities, wt_Hausdorf, save_path)
    print("=============")
    print('WT Dice: %.3f±%.3f' % (np.mean(wt_dices), np.std(wt_dices)))
    # print('TC Dice: %.3f±%.3f' % (np.mean(tc_dices), np.std(tc_dices)))
    # print('ET Dice: %.3f±%.3f' % (np.mean(et_dices), np.std(et_dices)))
    print("=============")
    print('WT sensitivity: %.3f±%.3f' % (np.mean(wt_sensitivities), np.std(wt_sensitivities)))
    # print('TC sensitivity: %.3f±%.3f' % (np.mean(tc_sensitivities), np.std(tc_sensitivities)))
    # print('ET sensitivity: %.3f±%.3f' % (np.mean(et_sensitivities), np.std(et_sensitivities)))
    print("=============")
    print('WT Hausdorff: %.3f±%.3f' % (np.mean(wt_Hausdorf), np.std(wt_Hausdorf)))
    # print('TC Hausdorff: %.4f' % np.mean(tc_Hausdorf))
    # print('ET Hausdorff: %.4f' % np.mean(et_Hausdorf))
    print("=============")
    save_csv(outputs_path, wt_dices, wt_sensitivities, wt_Hausdorf, save_path)


if __name__ == '__main__':
    main()


