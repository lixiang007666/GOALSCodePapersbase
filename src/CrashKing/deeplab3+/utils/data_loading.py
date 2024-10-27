import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .opencv_transforms import RandomCrop, RandomHorizontalFlip, RandomVerticalFlip


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask


class OCTDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, new_size=512):
        self.images_dir = Path(images_dir)
        if masks_dir is not None:
            self.masks_dir = Path(masks_dir)

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        self.new_size = new_size

        self.train_transform = DualCompose([
            RandomCrop(output_size=new_size),
            RandomHorizontalFlip(prob=0.5),
            RandomVerticalFlip(prob=0.5),
        ])

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def preprocess_train(self, pil_img, pil_mask):

        img_ndarray = np.asarray(pil_img).copy()
        mask_ndarray = np.asarray(pil_mask).copy()

        mask_ndarray2=np.zeros((4,mask_ndarray.shape[0],mask_ndarray.shape[1]))

        img_ndarray, mask_ndarray = self.train_transform(img_ndarray, mask_ndarray)

        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))
        img_ndarray = img_ndarray / 255

        mask_ndarray[mask_ndarray == 80] = 1
        mask_ndarray[mask_ndarray == 160] = 2
        mask_ndarray[mask_ndarray == 255] = 3
        mask_ndarray = mask_ndarray[:, :, 1].astype("int64")

        mask_ndarray2[0,mask_ndarray == 0] = 1
        mask_ndarray2[1,mask_ndarray==80]=1
        mask_ndarray2[2,mask_ndarray==160]=1
        mask_ndarray2[3,mask_ndarray==255]=1
        mask_ndarray2.transpose()
        mask_ndarray2=mask_ndarray2.astype('int64')


        return img_ndarray, mask_ndarray,mask_ndarray2

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img, mask,mask2 = self.preprocess_train(img, mask)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'mask2':torch.as_tensor(mask2.copy()).long().contiguous()
        }


class Inference_Dataset(Dataset):
    def __init__(self, image_dir, csv_file):
        self.image_dir = image_dir
        self.csv_file = pd.read_csv(csv_file, header=None)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        filename = self.csv_file.iloc[idx, 0]
        _, filename = os.path.split(filename)
        image_path = os.path.join(self.image_dir, filename)
        image = np.asarray(Image.open(image_path)).copy()  # mode:RGB

        image = self.transform(image)

        pos_list = self.csv_file.iloc[idx, 1:].values.astype(
            "int")  # ---> (topleft_x,topleft_y,buttomright_x,buttomright_y)

        return image, pos_list


if __name__ == '__main__':
    test_dir = '../../Validation/Image'  # 测试图像路径
    test_set = OCTDataset(test_dir, new_size=256, train=False)
    loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=True, **loader_args)

    for batch in test_loader:
        print(batch.shape)
