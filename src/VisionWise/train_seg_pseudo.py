import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import numpy as np
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import random
from torch.cuda.amp import autocast as autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts,StepLR, OneCycleLR
from torch.nn import CrossEntropyLoss

class OCT(BaseDataset):
	"""CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
	
	Args:
		images_dir (str): path to images folder
		masks_dir (str): path to segmentation masks folder
		class_values (list): values of classes to extract from segmentation mask
		augmentation (albumentations.Compose): data transfromation pipeline 
			(e.g. flip, scale, etc.)
		preprocessing (albumentations.Compose): data preprocessing 
			(e.g. noralization, shape manipulation, etc.)
	
	"""
	
	def __init__(self,images_dir, masks_dir,mode='train', augmentation=None, preprocessing=None,):
		self.imgfolder = images_dir
		self.labelfolder = masks_dir
		self.imgs = []
		self.labels = []
		self.mode = mode
		self.class_values = [0,1,2,3,4,5]
		selected_list = os.listdir(self.imgfolder)
		# random.seed(1)
		# random.shuffle(selected_list)
		# print(selected_list)
		if self.mode == 'train':
			for imgfile in os.listdir(self.imgfolder)[:90]:
				self.imgs.append(self.imgfolder + '/' + imgfile)
				self.labels.append(self.labelfolder + '/' + imgfile)
		elif self.mode == 'val':
			for imgfile in os.listdir(self.imgfolder)[90:]:
				self.imgs.append(self.imgfolder + '/' + imgfile)
				self.labels.append(self.labelfolder + '/' + imgfile)
		else:
			for imgfile in os.listdir(self.imgfolder):
				self.imgs.append(self.imgfolder + '/' + imgfile)
				self.labels.append(self.labelfolder + '/' + imgfile)
		
		if len(self.imgs) == 0:
			raise RuntimeError('Found 0 images, please check the data set')
		
		self.augmentation = augmentation
		self.preprocessing = preprocessing
	
	def __getitem__(self, i):
		
		# read data
		image = cv2.imread(self.imgs[i])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(self.labels[i], 0)
		
		# extract certain classes from mask (e.g. cars)
		masks = [(mask == v) for v in self.class_values]
		mask = np.stack(masks, axis=-1).astype('float')
		
		# apply augmentations
		if self.augmentation:
			sample = self.augmentation(image=image, mask=mask)
			image, mask = sample['image'], sample['mask']
		
		# apply preprocessing
		if self.preprocessing:
			sample = self.preprocessing(image=image, mask=mask)
			image, mask = sample['image'], sample['mask']
			
		return image, mask
		
	def __len__(self):
		return len(self.imgs)

def get_training_augmentation():
	train_transform = [

		albu.HorizontalFlip(p=0.5),
		# albu.ElasticTransform(),
		albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=20, shift_limit=0.1, p=1, border_mode=0),
		
		# albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
		albu.RandomCrop(height=768, width=768, always_apply=True),

		albu.GaussNoise(p=0.2),
		# albu.IAAPerspective(p=0.5),

		albu.OneOf(
			[
				albu.CLAHE(p=1),
				albu.RandomBrightness(p=1),
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
		albu.Resize(800,1088)
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

def visualize(image,mask):
	"""PLot images in one row."""
	fig,axes = plt.subplots(1,2)
	axes[0].imshow(image[0,:,:],cmap='gray')
	axes[0].axis('off')
	axes[1].imshow(np.argmax(mask,axis=0),cmap='jet',alpha=0.5)
	axes[1].axis('off')
	plt.tight_layout()
	plt.show()
	plt.close()


if __name__ == '__main__':
	# start = time.time
	x_train_dir = 'D:/GOALS2022-Train/Train/Image'
	y_train_dir = 'D:/GOALS2022-Train/Train/Masks'

	x_pseudo_dir = 'D:/GOALS_pseudo_3/Image'
	y_pseudo_dir = 'D:/GOALS_pseudo_3/Masks'
	
	# ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
	# 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d', 
	# 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 
	# 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 
	# 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 
	# 'densenet121', 'densenet169', 'densenet201', 'densenet161', 
	# 'inceptionresnetv2', 'inceptionv4', 
	# 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 
	# 'mobilenet_v2', 'xception', 
	# 'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2', 'timm-efficientnet-b3', 'timm-efficientnet-b4', 'timm-efficientnet-b5', 'timm-efficientnet-b6', 
	# 'timm-efficientnet-b7', 'timm-efficientnet-b8', 'timm-efficientnet-l2', 'timm-tf_efficientnet_lite0', 'timm-tf_efficientnet_lite1', 'timm-tf_efficientnet_lite2', 
	# 'timm-tf_efficientnet_lite3', 'timm-tf_efficientnet_lite4', 'timm-resnest14d', 'timm-resnest26d', 'timm-resnest50d', 'timm-resnest101e', 'timm-resnest200e', 
	# 'timm-resnest269e', 'timm-resnest50d_4s2x40d', 'timm-resnest50d_1s4x24d', 'timm-res2net50_26w_4s', 'timm-res2net101_26w_4s', 'timm-res2net50_26w_6s', 
	# 'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s', 'timm-res2net50_14w_8s', 'timm-res2next50', 'timm-regnetx_002', 'timm-regnetx_004', 'timm-regnetx_006', 
	# 'timm-regnetx_008', 'timm-regnetx_016', 'timm-regnetx_032', 'timm-regnetx_040', 'timm-regnetx_064', 'timm-regnetx_080', 'timm-regnetx_120', 'timm-regnetx_160', 
	# 'timm-regnetx_320', 'timm-regnety_002', 'timm-regnety_004', 'timm-regnety_006', 'timm-regnety_008', 'timm-regnety_016', 'timm-regnety_032', 'timm-regnety_040', 
	# 'timm-regnety_064', 'timm-regnety_080', 'timm-regnety_120', 'timm-regnety_160', 'timm-regnety_320', 'timm-skresnet18', 'timm-skresnet34', 'timm-skresnext50_32x4d', 
	# 'timm-mobilenetv3_large_075', 'timm-mobilenetv3_large_100', 'timm-mobilenetv3_large_minimal_100', 'timm-mobilenetv3_small_075', 'timm-mobilenetv3_small_100', 
	# 'timm-mobilenetv3_small_minimal_100', 'timm-gernet_s', 'timm-gernet_m', 'timm-gernet_l']
	
	ENCODER = 'timm-gernet_l'
	ENCODER_WEIGHTS = 'imagenet'
	ACTIVATION = None # could be None for logits or 'softmax2d' for multiclass segmentation
	DEVICE = 'cuda'

	# create segmentation model with pretrained encoder
	# model = smp.FPN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,classes=6,activation=ACTIVATION)
	model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,classes=6,activation=ACTIVATION,decoder_attention_type=None)
	# model = smp.PAN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,classes=6,activation=ACTIVATION)
	# model = smp.MAnet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,classes=6,activation=ACTIVATION)

	preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
	train_dataset = OCT(x_train_dir, y_train_dir, augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn),mode='train')
	pseudo_dataset = OCT(x_pseudo_dir, y_pseudo_dir, augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn),mode='pseudo')
	valid_dataset = OCT(x_train_dir, y_train_dir, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn),mode='val')

	train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)
	pseudo_loader = DataLoader(pseudo_dataset, batch_size=16, shuffle=True, num_workers=1)
	valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

	# id = random.randint(0,len(train_dataset))
	# for id in range(len(pseudo_dataset)):
	# 	image, mask = pseudo_dataset[id] # get some sample
	# 	visualize(
	# 		image=image, 
	# 		mask=mask,
	# 	)

	loss = [smp.losses.DiceLoss(mode='multilabel'),smp.losses.FocalLoss(mode='multilabel')]
	# loss = [smp.losses.DiceLoss(mode='multilabel'),smp.losses.SoftCrossEntropyLoss(smooth_factor=0.01)]
	# loss = [smp.losses.DiceLoss(mode='multilabel'),CrossEntropyLoss()]
	# loss = [smp.losses.DiceLoss(mode='multilabel'),smp.losses.LovaszLoss(mode='multilabel'),smp.losses.FocalLoss(mode='multilabel')]
	metrics = [smp.utils.metrics.Fscore(threshold=0.5)]

	
	base_lr = 0.0003
	optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=base_lr)])
	# optimizer = torch.optim.SGD([dict(params=model.parameters(), lr=base_lr)])
	# scheduler = OneCycleLR(optimizer,max_lr=0.001,epochs=200,steps_per_epoch=12,pct_start=0.3,verbose=False)
	scheduler = StepLR(optimizer,step_size=40,gamma=0.3,verbose=True)
	scaler = GradScaler()
	train_epoch = smp.utils.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optimizer,scheduler=scheduler,device=DEVICE,verbose=True,amp=True,scaler=scaler)
	pseudo_epoch = smp.utils.train.PseudoEpoch(model, loss=loss, weight=1,metrics=metrics, optimizer=optimizer,scheduler=scheduler,device=DEVICE,verbose=True,amp=True,scaler=scaler)
	valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=metrics, device=DEVICE,verbose=True)

	max_score = 0

	for i in range(0, 200):
		
		print('\nEpoch: {}'.format(i))
		train_logs = train_epoch.run(train_loader)
		pseudo_logs = pseudo_epoch.run(pseudo_loader)
		valid_logs = valid_epoch.run(valid_loader)
		scheduler.step()
		# do something (save model, change lr, etc.)
		if max_score < valid_logs['fscore']:
			max_score = valid_logs['fscore']
			torch.save(model, './unet_timm-gernet_l_pseudo3_best.pth')
			print('Model saved! Best:',max_score)
		print('LR:',optimizer.param_groups[0]['lr'])
		if (i+1) % 5 == 0:
			torch.save(model,'./unet_timm-gernet_l_pseudo3_ep' + str(i) + '.pth')
		