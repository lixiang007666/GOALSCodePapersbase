import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import numpy as np
import segmentation_models_pytorch as smp
import imageio

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import random

class OCT_Test_TTA(BaseDataset):
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
	
	def __init__(self,images_dir,mode='test', augmentation=None, preprocessing=None,):
		self.imgfolder = images_dir
		self.imgs = []
		self.labels = []
		self.mode = mode
		self.class_values = [0,1,2,3,4]
	
		# print(selected_list)
		for imgfile in os.listdir(self.imgfolder):
			self.imgs.append(self.imgfolder + '/' + imgfile)
			# self.labels.append(self.labelfolder + '/' + imgfile)
		
		if len(self.imgs) == 0:
			raise RuntimeError('Found 0 images, please check the data set')
		
		self.augmentation = get_validation_augmentation()
		self.preprocessing = preprocessing
	
	def __getitem__(self, i):
		
		# read data
		image = cv2.imread(self.imgs[i])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		imgs = []
		imgs.append(image)
		imgs.append(cv2.flip(image,1))
		imgs.append(gamma_trans(image,0.8))
		imgs.append(gamma_trans(image,1.2))
		states = ['orig','hflip' ,'g08','g12']
		# apply augmentations

		# fig,axes = plt.subplots(1,4)
		# axes[0].imshow(imgs[0],cmap='gray',vmin=0,vmax=255)
		# axes[0].axis('off')
		# axes[1].imshow(imgs[1],cmap='gray',vmin=0,vmax=255)
		# axes[1].axis('off')
		# axes[2].imshow(imgs[2],cmap='gray',vmin=0,vmax=255)
		# axes[2].axis('off')
		# axes[3].imshow(imgs[3],cmap='gray',vmin=0,vmax=255)
		# axes[3].axis('off')
		# plt.tight_layout()
		# plt.show()
		# plt.close()
		imgs_final = []
		
		for image in imgs:
			sample = self.augmentation(image=image)
			image = sample['image']
			
			# apply preprocessing
			if self.preprocessing:
				sample = self.preprocessing(image=image)
				image = sample['image']
			imgs_final.append(image)
			
		return imgs_final, states, self.imgs[i].split('/')[-1]
		
	def __len__(self):
		return len(self.imgs)

def gamma_trans(img, gamma):  # gamma函数处理
	gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
	gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
	return cv2.LUT(img, gamma_table)




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
	network_name = 'unet_timm-resnest101e_pseudo3_ep169'
	x_test_dir = 'D:/GOALS2022-Validation/Validation/Image'
	vis_folder = 'D:/PyDL/segmentation_models.pytorch-master/result/' + network_name 
	output_folder = 'D:/PyDL/segmentation_models.pytorch-master/output/' + network_name +'_tta_hflip'
	if not os.path.exists(vis_folder):
		os.mkdir(vis_folder)
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)
	
	ENCODER = 'timm-resnest101e'
	ENCODER_WEIGHTS = 'imagenet'
	ACTIVATION = None # could be None for logits or 'softmax2d' for multiclass segmentation
	DEVICE = 'cuda'

	# create segmentation model with pretrained encoder
	# model = smp.FPN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,classes=5,activation=ACTIVATION)
	best_model = torch.load('D:/PyDL/segmentation_models.pytorch-master/pseudo3/' + network_name + '.pth')

	preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
	test_dataset = OCT_Test_TTA(x_test_dir, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn),mode='test')

	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
	

	# id = random.randint(0,len(test_dataset))
	# image, mask = test_dataset[id] # get some sample
	# visualize(
	# 	image=image, 
	# 	mask=mask,
	# )

	for inputs, states, names in test_loader:
		prob_final = []
		for i in range(len(inputs)):
			input = inputs[i]
			state = states[i][0]
			x_tensor = input.to(DEVICE)
			prob_mask = best_model.predict(x_tensor)
			prob_mask = torch.softmax(prob_mask,dim=1).squeeze().cpu().numpy()
			if state == 'hflip':
				prob_mask = prob_mask[:,:,::-1]
			prob_final.append(prob_mask)
		
		prob_final = np.array(prob_final)
		prob_final = np.mean(prob_final,axis=0)
		prob_final = prob_final.astype(np.float16)

		np.save(output_folder+'/'+names[0].replace('.png','.npy'),prob_final)

		pred_mask = np.argmax(prob_final,axis=0)
		pred_mask = pred_mask.astype(np.uint8)
		imageio.imsave(output_folder + '/' + names[0], pred_mask)

		# plt.figure(figsize=(20,8))
		# fig,axes = plt.subplots(1,2)
		# img = input.detach().cpu().numpy()[0,0,:,:]
		# axes[0].imshow(img,cmap='gray')
		# axes[0].axis('off')
		# axes[1].imshow(img,cmap='gray')
		# axes[1].imshow(np.argmax(pr_mask,axis=0),cmap='jet',alpha=0.7)
		# axes[1].axis('off')
		# plt.tight_layout()
		# # plt.show()
		# plt.savefig(vis_folder + '/' + names[0])
		# plt.close()


	