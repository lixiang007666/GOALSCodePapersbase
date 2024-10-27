### 从数据文件夹中加载眼底图像，提取相应的金标准，生成训练样本


import math
import os
import cv2
import random
import h5py
import numpy as np
from PIL import Image, ImageDraw
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
from scipy.stats import multivariate_normal



class RandomGeneratorROI(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.transforms = transforms.Compose([
            transforms.Resize((self.output_size[0], self.output_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transforms_heatmap = transforms.Compose([
            transforms.Resize((self.output_size[0], self.output_size[1])),
            transforms.ToTensor(),
        ])

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = self.transforms(image)

        label = torch.Tensor(label)

        sample = {'image': image, 'label': label}
        # print(label)
        return sample



class OCTDataset(Dataset):
    def __init__(self, base_dir, split, transform=None):

        self.transform = transform  # using transform in torch!
        self.split = split
        self.data_dir = base_dir

        self.sample_list = open(os.path.join(self.data_dir, self.split, self.split + '.txt')).readlines()  ### list

        self.oct_images = []

        self.oct_masks = []

        self.oct_edges = []

        self.oct_RNFLs= []
        self.oct_GCIPLs = []
        self.oct_Choriods = []

        if self.split == "testing":
            begin=0
        else:
            begin=100

        index_len = len(self.sample_list)

        for k in range(index_len):
            print('Loading {} sample {}/{}...'.format(split, k, index_len), end='\r')

            # Image
            ImgName = str(k + 1 + begin).zfill(4)
            img_name = os.path.join(self.data_dir, self.split, 'Fundus_color_images', ImgName + '.png')
            # save_img_name = os.path.join(self.data_dir, self.split, 'Fundus_color_images_1', ImgName + '.png')
            img0  = cv2.imread(img_name)

            imgwidth = img0.shape[1]
            imgheight = img0.shape[0]
            print(imgwidth, imgheight)

            leftwidth1 = 0
            upperwidth1 = 50

            rightwidth1 = imgwidth
            underwidth1 = imgheight - 200

            img0 = img0[upperwidth1:underwidth1, leftwidth1:rightwidth1]



            # img = np.zeros([1100,1100,3])

            # img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

            # img0 = 255 * np.power(img0 / 255, 1.5)
            #
            # img0 = np.around(img0)
            #
            # img0[img0 > 255] = 255
            #
            # img0 = img0.astype(np.uint8)
            # cv2.imwrite(save_img_name, img0)


            # Seg
            seg_name = os.path.join(self.data_dir, self.split, 'Layer_Masks', ImgName + '.png')
            save_seg_name = os.path.join(self.data_dir, self.split, 'Layer_Masks_1', ImgName + '.png')
            seg0 = cv2.imread(seg_name)

            # rows, cols = img0.shape[:2]
            # # 简单理解：x方向移动100个单位，y方向移动50个单位
            # M = np.float32([[1, 0, 0], [0, 1, 0]])
            # # 输出图像大小
            # seg = cv2.warpAffine(seg0, M, (1100, 1100),borderValue=(255,255,255))
            # img = cv2.warpAffine(img0, M, (1100, 1100))

            # print(seg.shape,img.shape)

            # cv2.imwrite(save_img_name,img)
            # cv2.imwrite(save_seg_name,seg)

            seg = seg0
            img = img0

            # print(img.shape)

            edge_name = os.path.join(self.data_dir, self.split, 'edge', ImgName + '.png')
            edge = cv2.imread(edge_name)
            oct_edge = edge
            oct_edge[oct_edge == 255] = 1

            oct_edge = zoom(oct_edge, (896 / 1100, 448 / 550, 1), order=0)
            oct_edge = oct_edge[:, :, 1]

            oct_edge = torch.tensor(oct_edge.astype(np.float32))
            self.oct_edges.append(oct_edge)


            oct_mask = seg
            oct_mask[oct_mask == 80] = 1
            oct_mask[oct_mask == 160] = 2
            oct_mask[oct_mask == 255] = 3

            oct_mask = zoom(oct_mask, (1024 / 1100, 512 / 550, 1), order=0)

            oct_mask = oct_mask[:, :, 1]

            # print(oct_mask[224])

            # out_mask1 = np.zeros([800,1100])

            oct_RNFL = np.zeros_like(oct_mask)
            oct_GCIPL = np.zeros_like(oct_mask)
            oct_Choriod = np.zeros_like(oct_mask)

            oct_RNFL[oct_mask==0] = 1
            oct_GCIPL[oct_mask == 1] = 1
            oct_Choriod[oct_mask == 2] = 1

            oct_RNFL = torch.tensor(oct_RNFL.astype(np.float32))
            oct_GCIPL = torch.tensor(oct_GCIPL.astype(np.float32))
            oct_Choriod = torch.tensor(oct_Choriod.astype(np.float32))

            self.oct_RNFLs.append(oct_RNFL)
            self.oct_GCIPLs.append(oct_GCIPL)
            self.oct_Choriods.append(oct_Choriod)

            # print(img.shape)
            # img = cv2.resize(img, (1024, 512))

            # print(img[1])
            # print(img.shape)
            # print(img.shape)
            # print(img.shape,oct_mask.shape)

            self.oct_images.append(img)
            self.oct_masks.append(oct_mask)


        print('Succesfully loaded {} dataset.'.format(split) + '*' * 50)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img0 = self.oct_images[idx]
        oct_mask = self.oct_masks[idx]

        r = random.uniform(0.0, 1.0)

        if r > 0.5 and self.split!='testing':
            img0 = cv2.flip(img0, 1)
            # print(oct_mask0.shape)
            oct_mask = cv2.flip(oct_mask, 1)
            # print(oct_mask0.shape)


        ## 随机裁剪放缩
        ratio = random.uniform(0.5, 1.5)
        r3 = random.uniform(0.0, 1.0)

        if r3 > 0.5 and self.split != "testing":

            img_h, img_w = img0.shape[0], img0.shape[1]

            crop_h = img_h
            crop_w = img_w

            #	裁剪比例
            scale = ratio
            height, width = int(img_h * scale), int(img_w * scale)

            img0 = cv2.resize(img0,(width, height))
            oct_mask = cv2.resize(oct_mask,(width, height), interpolation=cv2.INTER_NEAREST)

            if(height<img_h):

                top = random.randint(0, (img_h - height))
                bottom = (img_h - height)  - top
                left = random.randint(0, (img_w - width))
                right =  (img_w - width)  - left

                img0 = cv2.copyMakeBorder(img0, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
                oct_mask = cv2.copyMakeBorder(oct_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(3))

            #	随机生成裁剪区域的起点
            img_h1, img_w1 = img0.shape[0], img0.shape[1]

            x = random.randint(0, img_w1 - crop_w)
            y = random.randint(0, img_h1 - crop_h)

            #	在图像的原区域进行裁剪，取相应区域的像素
            img0 = img0[y:y + img_h, x:x + img_w]
            oct_mask = oct_mask[y:y + img_h, x:x + img_w]


        (h, w) = img0.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        r1 = random.uniform(0.0, 1.0)
        num = random.randint(-30, 30)

        if r1 > 0.5 and self.split != 'testing':

            M = cv2.getRotationMatrix2D((cX, cY), num, 1.0)

            img0 = cv2.warpAffine(img0, M, (w, h))
            # print(oct_mask0.shape)
            oct_mask = cv2.warpAffine(oct_mask, M, (w, h), borderValue=(3))
            # print(oct_mask0.shape)


        leftnum = random.randint(-50, 50)
        topnum = random.randint(-100, 100)
        r2 = random.uniform(0.0, 1.0)


        if r2 > 0.5 and self.split != "testing":
            rows, cols = img0.shape[:2]
            # 简单理解：x方向移动100个单位，y方向移动50个单位
            M = np.float32([[1, 0, leftnum], [0, 1, topnum]])
            img0 = cv2.warpAffine(img0, M, (cols, rows))
            oct_mask = cv2.warpAffine(oct_mask, M, (w, h), borderValue=(3))

        img = img0.transpose(2, 0, 1)
        img = torch.tensor(img.astype(np.float32))

        oct_RNFL = np.zeros_like(oct_mask)
        oct_GCIPL = np.zeros_like(oct_mask)
        oct_Choriod = np.zeros_like(oct_mask)

        oct_RNFL[oct_mask == 0] = 1
        oct_GCIPL[oct_mask == 1] = 1
        oct_Choriod[oct_mask == 2] = 1

        oct_RNFL = torch.tensor(oct_RNFL.astype(np.float32))
        oct_GCIPL = torch.tensor(oct_GCIPL.astype(np.float32))
        oct_Choriod = torch.tensor(oct_Choriod.astype(np.float32))

        oct_mask = torch.tensor(oct_mask.astype(np.float32))

        sample = {'oct_image': img,
                  'oct_mask': oct_mask,
                  'oct_RNFL': oct_RNFL,
                  'oct_GCIPL': oct_GCIPL,
                  'oct_Choriod': oct_Choriod,
                  'oct_edge': self.oct_edges[idx],
                  }

        sample['case_name'] = self.sample_list[idx].strip('\n').zfill(4)

        return sample

class OCTDataset_noargument(Dataset):
    def __init__(self, base_dir, split, transform=None):

        self.transform = transform  # using transform in torch!
        self.split = split
        self.data_dir = base_dir

        self.sample_list = open(os.path.join(self.data_dir, self.split, self.split + '.txt')).readlines()  ### list

        self.oct_images = []

        self.oct_masks = []

        self.oct_edges = []

        self.oct_RNFLs= []
        self.oct_GCIPLs = []
        self.oct_Choriods = []

        if self.split == "testing":
            begin=0
        else:
            begin=100

        index_len = len(self.sample_list)

        for k in range(index_len):
            print('Loading {} sample {}/{}...'.format(split, k, index_len), end='\r')

            # Image
            ImgName = str(k + 1 + begin).zfill(4)
            img_name = os.path.join(self.data_dir, self.split, 'Fundus_color_images', ImgName + '.png')
            save_img_name = os.path.join(self.data_dir, self.split, 'Fundus_color_images_1', ImgName + '.png')
            img0  = cv2.imread(img_name)



            # Seg
            seg_name = os.path.join(self.data_dir, self.split, 'Layer_Masks', ImgName + '.png')
            save_seg_name = os.path.join(self.data_dir, self.split, 'Layer_Masks_1', ImgName + '.png')
            seg0 = cv2.imread(seg_name)

            seg = seg0
            img = img0

            edge_name = os.path.join(self.data_dir, self.split, 'edge', ImgName + '.png')
            edge = cv2.imread(edge_name)
            oct_edge = edge
            oct_edge[oct_edge == 255] = 1

            # oct_edge = zoom(oct_edge, (896 / 1100, 448 / 550, 1), order=0)
            oct_edge = oct_edge[:, :, 1]

            oct_edge = torch.tensor(oct_edge.astype(np.float32))
            self.oct_edges.append(oct_edge)


            oct_mask = seg
            oct_mask[oct_mask == 80] = 1
            oct_mask[oct_mask == 160] = 2
            oct_mask[oct_mask == 255] = 3

            # oct_mask = zoom(oct_mask, (1024 / 1100, 512 / 550, 1), order=0)

            oct_mask = oct_mask[:, :, 1]



            # print(img.shape)
            # img = cv2.resize(img, (1024, 512))

            self.oct_images.append(img)
            self.oct_masks.append(oct_mask)


        print('Succesfully loaded {} dataset.'.format(split) + '*' * 50)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img0 = self.oct_images[idx]
        oct_mask = self.oct_masks[idx]

        img = img0.transpose(2, 0, 1)
        img = torch.tensor(img.astype(np.float32))

        oct_RNFL = np.zeros_like(oct_mask)
        oct_GCIPL = np.zeros_like(oct_mask)
        oct_Choriod = np.zeros_like(oct_mask)

        oct_RNFL[oct_mask == 0] = 1
        oct_GCIPL[oct_mask == 1] = 1
        oct_Choriod[oct_mask == 2] = 1

        oct_RNFL = torch.tensor(oct_RNFL.astype(np.float32))
        oct_GCIPL = torch.tensor(oct_GCIPL.astype(np.float32))
        oct_Choriod = torch.tensor(oct_Choriod.astype(np.float32))

        oct_mask = torch.tensor(oct_mask.astype(np.float32))

        sample = {'oct_image': img,
                  'oct_mask': oct_mask,
                  'oct_RNFL': oct_RNFL,
                  'oct_GCIPL': oct_GCIPL,
                  'oct_Choriod': oct_Choriod,
                  'oct_edge': self.oct_edges[idx],
                  }

        sample['case_name'] = self.sample_list[idx].strip('\n').zfill(4)

        return sample