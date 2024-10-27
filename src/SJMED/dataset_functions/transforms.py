import albumentations
import albumentations.augmentations.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2
import torch
import cv2
from torchvision.transforms.transforms import CenterCrop
import numpy as np

def img_transforms(New_size = (800,800), applied_types = 'train'):
	## albumentations
	## can be used as basic traning and test set, without any augmentation
	if applied_types == None:
		data_transforms = albumentations.Compose([
		    albumentations.Resize(New_size[0], New_size[1]),
		    ToTensorV2()
		    ])

	elif applied_types == "train":
		data_transforms = albumentations.Compose([
			albumentations.Resize(New_size[0], New_size[1]),
			albumentations.RandomResizedCrop(height = New_size[0], width = New_size[1], scale=(0.95, 1.05), ratio=(0.95, 1.05), p=0.25),
            albumentations.HorizontalFlip(p=0.5),
			albumentations.VerticalFlip(p=0.25),
			albumentations.ShiftScaleRotate(shift_limit=0.0625,
										scale_limit=0.05,
										rotate_limit=15,
										p=0.25),
			albumentations.OneOf([
				albumentations.Blur(blur_limit=5),
				albumentations.GaussianBlur(blur_limit=5),
				albumentations.MedianBlur(blur_limit=5),
				albumentations.MotionBlur(blur_limit=5)
				], p=0.25),
			transforms.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.25),
			albumentations.CoarseDropout(p=0.1),
			ToTensorV2()])
            
	elif applied_types == "val" or applied_types == "test":
		data_transforms = albumentations.Compose([
			albumentations.Resize(New_size[0], New_size[1]),
			ToTensorV2()
			])

	return data_transforms


def img_transforms_layermask(New_size = (800,800), applied_types = 'train'):
	## albumentations
	## can be used as basic traning and test set, without any augmentation
	if applied_types == None:
		data_transforms = albumentations.Compose([
		    albumentations.Resize(New_size[0], New_size[1]),
		    ToTensorV2()
		    ])

	elif applied_types == "train":
		data_transforms = albumentations.Compose([
			albumentations.Resize(New_size[0], New_size[1]),
			# albumentations.RandomResizedCrop(height = New_size[0], width = New_size[1], scale=(0.95, 1.05), ratio=(0.95, 1.05), p=0.25),
            # albumentations.HorizontalFlip(p=0.5),
			# albumentations.VerticalFlip(p=0.25),
			# albumentations.ShiftScaleRotate(shift_limit=0.0625,
			# 							scale_limit=0.05,
			# 							rotate_limit=15,
			# 							p=0.25),
			albumentations.OneOf([
				albumentations.Blur(blur_limit=5),
				albumentations.GaussianBlur(blur_limit=5),
				albumentations.MedianBlur(blur_limit=5),
				albumentations.MotionBlur(blur_limit=5)
				], p=0.25),
			transforms.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.25),
			# albumentations.CoarseDropout(p=0.1),
			ToTensorV2()])
            
	elif applied_types == "val" or applied_types == "test":
		data_transforms = albumentations.Compose([
			albumentations.Resize(New_size[0], New_size[1]),
			ToTensorV2()
			])

	return data_transforms


def img_transforms_more(New_size = (800,800), applied_types = 'train'):
	## albumentations
	## can be used as basic traning and test set, without any augmentation
	if applied_types == None:
		data_transforms = albumentations.Compose([
		    albumentations.Resize(New_size[0], New_size[1]),
		    ToTensorV2()
		    ])

	elif applied_types == "train":
		data_transforms = albumentations.Compose([
			albumentations.Resize(New_size[0], New_size[1]),
			albumentations.RandomResizedCrop(height = New_size[0], width = New_size[1], scale=(0.95, 1.05), ratio=(0.95, 1.05), p=0.25),
            albumentations.HorizontalFlip(p=0.5),
			albumentations.VerticalFlip(p=0.25),
			albumentations.ShiftScaleRotate(shift_limit=0.0625,
										scale_limit=0.05,
										rotate_limit=15,
										p=0.25),
			albumentations.OneOf([
				albumentations.Blur(blur_limit=5),
				albumentations.GaussianBlur(blur_limit=5),
				albumentations.MedianBlur(blur_limit=5),
				albumentations.MotionBlur(blur_limit=5)
				], p=0.25),
			transforms.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.25),
			albumentations.CoarseDropout(p=0.1),
			albumentations.CLAHE(p=0.2), # 直方图均衡化
			albumentations.Downscale(scale_min=0.8, scale_max=0.9, p=0.1),
			albumentations.GridDistortion(p=0.1),
			albumentations.MultiplicativeNoise(per_channel=True,elementwise=True,p=0.1),
			albumentations.RandomFog(fog_coef_lower=0.3,fog_coef_upper=0.4,p=0.1),
			albumentations.RandomGridShuffle(p=0.1),
			albumentations.RandomSnow(snow_point_lower=0.1,snow_point_upper=0.2,brightness_coeff=2,p=0.1),
			ToTensorV2()])
            
	elif applied_types == "val" or applied_types == "test":
		data_transforms = albumentations.Compose([
			albumentations.Resize(New_size[0], New_size[1]),
			ToTensorV2()
			])

	return data_transforms