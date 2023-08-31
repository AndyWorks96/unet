import SimpleITK as sitk

# imagePath = './data/BraTS20_Training_001_flair.dcm'
imagePath = './data/dcm/liu xiu wen/T2/1.2.840.113619.2.353.2807.8093420.14407.1468978603.343.dcm'
# imagePath = './data/dwiTest/cao yupu/cao yupu_dwi.nii.gz'
input_image = sitk.ReadImage(imagePath)
pixel_array = sitk.GetArrayFromImage(input_image)

input_image = sitk.Cast(input_image, sitk.sitkFloat32)
# print(pixel_array[70][155])
# mask_iamge = sitk.OtsuThresholdImageFilter(input_image,0,1,2,4,200)
corrector = sitk.N4BiasFieldCorrectionImageFilter()
output_image = corrector.Execute(input_image)
output_image = sitk.Cast(output_image,sitk.sitkInt16)
sitk.WriteImage(output_image,'./data/T2_343.nii.gz')