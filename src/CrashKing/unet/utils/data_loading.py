import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .opencv_transforms import RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from .data_augmentation import oct_data_aug


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str = None, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        if masks_dir is not None:
            self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask


class OCTDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir=None, new_size=512, type='train', transform=None, data_aug=False):
        super().__init__(images_dir, masks_dir)

        self.new_size = new_size

        self.type = type

        self.data_aug = data_aug

        if self.type == 'train':
            self.transform = DualCompose([
                RandomCrop(output_size=new_size),
                RandomHorizontalFlip(prob=0.5),
                RandomVerticalFlip(prob=0.5),
            ])
        elif self.type == 'val':
            self.transform = DualCompose([
                RandomCrop(output_size=new_size),
            ])
        elif self.type == 'test':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            raise NotImplementedError

    def preprocess_test(self, pil_img):

        img_ndarray = np.asarray(pil_img).copy()

        img_tensor = self.transform(img_ndarray)

        return img_tensor

    def preprocess_train_val(self, pil_img, pil_mask):

        img_ndarray = np.asarray(pil_img).copy()
        mask_ndarray = np.asarray(pil_mask).copy()

        if not self.data_aug:
            img_ndarray, mask_ndarray = self.transform(img_ndarray, mask_ndarray)
        else:
            img_ndarray, mask_ndarray = oct_data_aug(img_ndarray, mask_ndarray, self.new_size)

        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))
        img_ndarray = img_ndarray / 255

        mask_ndarray[mask_ndarray == 80] = 1
        mask_ndarray[mask_ndarray == 160] = 2
        mask_ndarray[mask_ndarray == 255] = 3
        mask_ndarray = mask_ndarray[:, :, 1].astype("int64")

        return img_ndarray, mask_ndarray

    def __getitem__(self, idx):
        if self.type in ['train', 'val']:
            name = self.ids[idx]
            mask_file = list(self.masks_dir.glob(name + '.*'))
            img_file = list(self.images_dir.glob(name + '.*'))

            assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
            assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
            mask = self.load(mask_file[0])
            img = self.load(img_file[0])

            assert img.size == mask.size, \
                f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

            img, mask = self.preprocess_train_val(img, mask)

            return {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous()
            }
        else:
            name = self.ids[idx]
            img_file = list(self.images_dir.glob(name + '.*'))

            assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
            img = self.load(img_file[0])

            img_tensor = self.preprocess_test(img)

            return img_tensor


class Inference_Dataset(Dataset):
    def __init__(self, image_dir, csv_file):
        '''
        Description:
        Args (type):
            csv_file  (string): Path to the file with annotations, see `utils/data_prepare` for more information.
            image_dir (string): Derectory with all images.
            transforms (callable,optional): Optional transforms to be applied on a sample.
        return:
        '''
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
    test_dir = '../../Validation/Image'
    test_set = OCTDataset(test_dir, new_size=256, type='train')
    loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=True, **loader_args)

    for batch in test_loader:
        print(batch.shape)
