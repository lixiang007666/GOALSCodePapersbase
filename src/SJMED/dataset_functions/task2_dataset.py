# task2自己定义的dataset
import cv2
import os
import pandas as pd
from torch import long
from torch.utils.data import Dataset

# 数据加载（继承torch.utils.data中的Dataset类）
# 输入：dataset_root为图片路径，filelists为存储图片文件名的列表，label_file为存储label的文件
# 输出：若mode=='tain' or 'val'输出imgs和labels，若mode=='test'输出imgs(test模式下图片没有标签)
class GOALS_sub2_dataset(Dataset):
    def __init__(self, img_transforms, dataset_root, filelists=None, label_file=None, mode='train'): 
        self.img_transforms = img_transforms
        self.dataset_root = dataset_root
        self.filelists = filelists
        self.label_file = label_file
        self.mode = mode

    def __getitem__(self, idx):
        if self.filelists == None:
            self.filelists = os.listdir(self.dataset_root)
        
        if self.mode == 'train' or self.mode == 'val': # 训练
            label_dict = {row['ImgName']:row['GC_Label'] for _, row in pd.read_excel(self.label_file).iterrows()} # label是一个图片对应label的字典
            img_index = int(self.filelists[idx].split('.')[0])
            label = label_dict[img_index]
            img = cv2.imread(os.path.join(self.dataset_root, self.filelists[idx]))
            # img = torch.from_numpy(img)/255
            img = img/255
        if self.mode == 'test': # 测试
            real_index = int(self.filelists[idx].split('.')[0])
            img = cv2.imread(os.path.join(self.dataset_root, self.filelists[idx]))
            # img = torch.from_numpy(img)/255
            img = img/255

        if self.img_transforms is not None:
            img = self.img_transforms(img)

        if self.mode == 'train' or self.mode == 'val':
            return img.float(), label
        if self.mode == 'test':
            return img.float(), real_index
    
    def __len__(self):
        return len(self.filelists)