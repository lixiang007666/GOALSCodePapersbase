import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd


def getSplitData(path_root, mode='train', train_valid_ratio=0.1):
    """
    Split data to train set and valid set
    train : valid = 9 : 1
    :param path_root: data path
    :param dataset: main dataset of experiment
    :param train_valid_ratio
    :return: img list and label list
    """
    path = None
    if mode == 'train':
        # data path
        path = path_root + 'Train/Image/'
        # data list
        list_oct = os.listdir(path)  # 100
        # print message
        print('Total read: {} images'.format(len(list_oct)))
        # train-valid
        train_list, valid_list = train_test_split(list_oct, test_size=train_valid_ratio, random_state=42)
        print('=' * 50)
        return train_list, valid_list
    elif mode == 'test':
        # data path
        path = path_root + 'Validation/Image/'
        # data list
        list_oct = os.listdir(path)  # 100
        # print message
        print('Total read: {} images'.format(len(list_oct)))
        print('=' * 50)
        return list_oct


class ImageSet(Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        self.data_dir = data_dir
        self.mode = mode.lower()
        self.transform = transform
        if self.mode == 'train':
            self.filelists, _ = getSplitData(path_root=self.data_dir, mode='train')
            self.filelabels = {row['ImgName']: row[1] for _, row in
                               pd.read_excel(data_dir + "Train/Train_GC_GT.xlsx").iterrows()}
        if self.mode == 'valid':
            _, self.filelists = getSplitData(path_root=self.data_dir, mode='train')
            self.filelabels = {row['ImgName']: row[1] for _, row in
                               pd.read_excel(data_dir + "Train/Train_GC_GT.xlsx").iterrows()}
        if self.mode == 'test':
            self.filelists = getSplitData(path_root=self.data_dir, mode='test')
            self.filelabels = None

    def __len__(self):
        self.file_length = len(self.filelists)
        return self.file_length

    def __getitem__(self, index):
        img, label = None, []
        file_name = self.filelists[index]
        if self.mode == 'train' or self.mode == 'valid':
            img = Image.open(self.data_dir + 'Train/Image/' + file_name)
            label = self.filelabels[int(file_name.split('.')[0])]
        elif self.mode == 'test':
            img = Image.open(self.data_dir + 'Validation/Image/' + file_name)
        else:
            pass
        # after ToTensor(), [0, 255] --> [0.0, 1.0]
        # original size is (1100, 800)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, file_name

