import skimage.io as io
from nilearn.image import resample_img
import os
import numpy as np
import SimpleITK as sitk

img_folder = r'G:\MICCAI\Train\train_seg\train\Image'
lab_folder = r'G:\MICCAI\Train\train_seg\train\Layer_Masks'


save_img = r'G:\MICCAI\Train\train_seg\train)resize\Image'
save_lan = r'G:\MICCAI\Train\train_seg\train)resize\Layer_Masks'



def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    # print('originSize',originSize)
    # originSize = itkimage.shape
    originSpacing = itkimage.GetSpacing()
    # print('originSpacing',originSpacing)
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    # print('newSpacing',newSpacing)
    newSize = newSize.astype(np.int)  # spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像

    return itkimgResampled


for imgfile in os.listdir(img_folder):
    imgpath = img_folder + '/' + imgfile
    img = sitk.ReadImage(imgpath)
    img = resize_image_itk(img, (1120, 800), sitk.sitkLinear)

    labpath = lab_folder + '/' + imgfile
    lab = sitk.ReadImage(labpath)
    lab = resize_image_itk(lab, (800, 1120), sitk.sitkNearestNeighbor)


    out_imgfile = os.path.join(save_img, imgfile.split('.')[0] + '.png')
    sitk.WriteImage(img, out_imgfile)

    out_labfile = os.path.join(save_lan, imgfile.split('.')[0] + '.png')
    sitk.WriteImage(lab, out_labfile)
