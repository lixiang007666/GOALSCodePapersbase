import os
import cv2
import torch
from torch.utils.data import Dataset


# 输入：image_path为数据集路径，filelists为此次调用OCTDataset的图像列表（训练或者验证的图像列表），gt_path为分割的ground truth
class OCTDataset(Dataset):
    def __init__(self, image_transforms, image_path, filelists=None, gt_path=None, mode='train'):
        super(OCTDataset, self).__init__()
        self.image_transforms = image_transforms
        self.image_path = image_path
        self.filelists = filelists
        if self.filelists == None:
            self.filelists = os.listdir(self.image_path)
        self.gt_path = gt_path
        self.mode = mode

    def __getitem__(self, idx):
        img_index = self.filelists[idx]
        img_path = os.path.join(self.image_path, img_index)
        img = cv2.imread(img_path)
        h, w, c = img.shape
        if self.mode == 'train' or self.mode == 'val':
            gt_img = cv2.imread(os.path.join(self.gt_path, img_index))
            
            # 像素值为0的是RNFL(类别 0)，像素值为80的是GCIPL(类别 1)，像素值为160的是脉络膜(类别 2)，像素值为255的是其他（类别3）
            gt_img[gt_img == 80] = 1
            gt_img[gt_img == 160] = 2
            gt_img[gt_img == 255] = 3
            
            gt_img = cv2.resize(gt_img, (256, 256))
            gt_img = torch.from_numpy(gt_img)
            gt_img = gt_img[:,:,1] # 取一个通道

            
                # gt_img = self.image_transforms(gt_img)
                # gt_img = torch.squeeze(gt_img, 0)

        # img = cv2.resize(img, (self.image_size, self.image_size))
        img = img/255

        if self.image_transforms is not None:
                img = self.image_transforms(img)

        if self.mode == 'train' or self.mode == 'val':
            return img.float(), gt_img.long()
        if self.mode == 'test':
            return img.float(), img_index, h, w

    def __len__(self):
        return len(self.filelists)