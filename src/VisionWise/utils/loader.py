# baldder.py
import os
# import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from skimage.transform import resize,rescale
from skimage.filters import scharr,scharr_h,scharr_v,sobel
from skimage.restoration import denoise_wavelet
from skimage.filters import gaussian
import random
import logging
import json
from skimage.exposure import rescale_intensity, adjust_gamma
from skimage.util import random_noise
import pandas as pd
import torch.nn.functional as F
import csv
import albumentations as albu
import cv2


palette = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]

num_classes = 10

def get_training_augmentation():
	train_transform = [

		albu.HorizontalFlip(p=0.5),
		# albu.ElasticTransform(),
		albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
		
		# albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
		# albu.RandomResizedCrop(height=512, width=512, scale=(0.8,1), ratio=(0.99,1.01),always_apply=True),
        albu.Resize(height=512, width=512,always_apply=True),

		albu.GaussNoise(p=0.2),
		# albu.IAAPerspective(p=0.5),

		albu.OneOf(
			[
				albu.CLAHE(p=1),
				albu.RandomBrightnessContrast(p=1),
				albu.RandomGamma(p=1),
			],
			p=0.9,
		),

		albu.OneOf(
			[
				albu.Sharpen(p=1),
				albu.Blur(blur_limit=3, p=1),
				albu.MotionBlur(blur_limit=3, p=1),
			],
			p=0.9,
		),

		albu.OneOf(
			[
				albu.RandomBrightnessContrast(p=1),
				albu.HueSaturationValue(p=1),
			],
			p=0.9,
		),
	]
	return albu.Compose(train_transform)


def get_validation_augmentation():
	"""Add paddings to make image shape divisible by 32"""
	test_transform = [
		# albu.PadIfNeeded(800, 1120),
		albu.Resize(512,512)
	]
	return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
	return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
	"""Construct preprocessing transform
	
	Args:
		preprocessing_fn (callbale): data normalization function 
			(can be specific for each pretrained neural network)
	Return:
		transform: albumentations.Compose
	
	"""
	
	_transform = [
		albu.Lambda(image=preprocessing_fn),
		albu.Lambda(image=to_tensor, mask=to_tensor),
	]
	return albu.Compose(_transform)

def preprocess_input(x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs):
    if input_space == "BGR":
        x = x[..., ::-1].copy()
    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0
    if mean is not None:
        mean = np.array(mean)
        x = x - mean
    if std is not None:
        std = np.array(std)
        x = x / std
    return x

class GLC(data.Dataset):
    def __init__(self, root, mode):
        self.imgfolder = root + '/Image'
        self.labelfolder = root + '/Masks'
        self.glclabel = root + '/GC_GT.csv'
        label_dict = {}
        with open(self.glclabel,'r',newline='') as f:
            reader = csv.reader(f,delimiter=',')
            next(reader)
            for row in reader:
                label_dict[row[0]] = int(row[1])
        self.imgs = []
        self.labels = []
        self.glclabels = []
        self.mode = mode
        if self.mode == 'train':
            for imgfile in os.listdir(self.imgfolder)[40:]:
                self.imgs.append(self.imgfolder + '/' + imgfile)
                self.labels.append(self.labelfolder + '/' + imgfile)
                id = imgfile.split('.')[0]
                id = str(int(id))
                self.glclabels.append(label_dict[id])
        else:
            for imgfile in os.listdir(self.imgfolder)[:40]:
                self.imgs.append(self.imgfolder + '/' + imgfile)
                self.labels.append(self.labelfolder + '/' + imgfile)
                id = imgfile.split('.')[0]
                id = str(int(id))
                self.glclabels.append(label_dict[id])
        
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        

    def __getitem__(self, index):
        img_path = self.imgs[index]
        mask_path = self.labels[index]
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        glc = self.glclabels[index]
        img = np.array(img)[:,:,0]
        img = rescale_intensity(img,out_range=np.uint8)

        # if self.joint_transform is not None:
        #     img, mask = self.joint_transform(img, mask)
        # if self.center_crop is not None:
        #     img, mask = self.center_crop(img, mask)
        
        mask = np.array(mask)

        if self.mode == 'train':
            #img = torch.from_numpy(img).float()
            #mask = torch.from_numpy(mask)
        
            #return img, mask, img_path

            # print(mask.max())

            # random horizontal flip
            flip_prob = random.random()
            if flip_prob < 0.5 and self.mode == 'train':
                img = np.fliplr(img)
                mask = np.fliplr(mask)

            # random vertical translate and scaling
            # random crop a region (height=256) which contains the full retinal layers
            trans_prob = random.random()
            if trans_prob < 0.5 and self.mode == 'train':
                upbound = np.argwhere(mask==1)
                lowbound = np.argwhere(mask==4)
                margin = 0
                miny = max(margin, np.min(upbound[:,0]) - margin)
                maxy = min(mask.shape[0], np.max(lowbound[:,0]) + margin)
                # print(maxy,miny)
                trans_range = [-miny,img.shape[0]-maxy]
                trans_y = random.randint(trans_range[0],trans_range[1])
                # print(trans_y)

                if trans_y < 0:
                    trans_y = abs(trans_y)
                    img_trans = np.zeros(img.shape)
                    mask_trans = np.ones(img.shape) * 5
                    img_trans[0:img.shape[0]-trans_y] = img[trans_y:]
                    mask_trans[0:img.shape[0]-trans_y] = mask[trans_y:]
                elif trans_y > 0:
                    img_trans = np.zeros(img.shape)
                    mask_trans = np.zeros(img.shape) 
                    img_trans[trans_y:] = img[0:img.shape[0]-trans_y]
                    mask_trans[trans_y:] = mask[0:img.shape[0]-trans_y]
                else:
                    img_trans = img
                    mask_trans = mask
                
                img = img_trans
                mask = mask_trans

            if random.random() < 0.5 and self.mode == 'train':
                gamma = 0.7 + random.random()
                img = adjust_gamma(img,gamma=gamma)
                img = rescale_intensity(img,out_range=np.uint8)
            
        img = resize(img,output_shape=(512,512),order=1,preserve_range=True)
        mask = resize(mask,output_shape=(512,512),order=0,preserve_range=True)

        img = np.expand_dims(img, axis=0).copy()
        mask = mask.copy()
        
        # print(mask[:, 1, 0])
        # shape from (H, W, C) to (C, H, W)
        
        ## check if augmentation is correct 
        # mask_back = np.argmax(mask,axis=0)
        # mask_bds = find_boundaries(mask_back,mode='inner')
        # plt.imshow(img[0,:,:],cmap='gray')
        # plt.imshow(mask_bds,alpha=0.5)
        # plt.imshow(mask_back,alpha=0.3)
        # plt.show()
        # plt.close()

        img = torch.from_numpy(img).float()
        img = img.repeat(3,1,1)

        mask = np.expand_dims(mask,axis=0).copy()
        mask = torch.from_numpy(mask).float()
        mask = mask.repeat(3,1,1)
        # mask = F.one_hot(mask,num_classes=6)
        # mask = mask.permute(2,0,1)
        
        return img, mask, glc, img_path

    def __len__(self):
        return len(self.imgs)


class GLC_Test(data.Dataset):
    def __init__(self, root, mode):
        self.imgfolder = root + '/Image'
        self.imgs = []
        self.mode = mode
        for imgfile in os.listdir(self.imgfolder):
            self.imgs.append(self.imgfolder + '/' + imgfile)
        
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        

    def __getitem__(self, index):
        img_path = self.imgs[index]
        
        img = Image.open(img_path)
        img = rescale_intensity(img,out_range=np.uint8)
        img = np.array(img)[:,:,0]
        img = resize(img,output_shape=(512,512),order=1,preserve_range=True)
        img = np.expand_dims(img, axis=0).copy()
        img = torch.from_numpy(img).float()
        img = img.repeat(3,1,1)
        
        return img, img_path

    def __len__(self):
        return len(self.imgs)
