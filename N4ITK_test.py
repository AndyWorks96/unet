import SimpleITK as sitk

# imagePath = './data/BraTS20_Training_001_flair.dcm'
imagePath = './data/BraTS20_Training_001_flair.dcm'
input_image = sitk.ReadImage(imagePath)
input_image = sitk.Cast(input_image, sitk.sitkFloat32)

pixel_array_pro = sitk.GetArrayFromImage(input_image)
print(pixel_array_pro[70][155])

# mask_iamge = sitk.OtsuThresholdImageFilter(input_image,0,1,2,4,200)
corrector = sitk.N4BiasFieldCorrectionImageFilter()
output_image = corrector.Execute(input_image)
output_image = sitk.Cast(output_image,sitk.sitkUInt8)
# 对处理后的数据进行像素值打印
pixel_array_pre = sitk.GetArrayFromImage(output_image)
print(pixel_array_pre[70][155])

sitk.WriteImage(output_image,'./data/test01.nii.gz')