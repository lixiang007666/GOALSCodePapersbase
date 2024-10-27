import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from PIL import Image
import torch
import albumentations as albu
import segmentation_models_pytorch as smp
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch.utils


class Dataset(BaseDataset):
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

    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']

    def __init__(
            self,
            images_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(self.CLASSES[cls].lower()) for cls in range(classes)]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image,self.images_fps[i].split('/')[-1]

    def __len__(self):
        return len(self.ids)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(800, 1120)
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
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)

ENCODER = 'timm-resnest101e'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = None
DEVICE = 'cuda'
x_test_dir='G://MICCAI//Image_resize'
best_model_path='G:/MICCAI/log_seg/best_model.pth'
best_model = torch.load(best_model_path)
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
test_dataset = Dataset(x_test_dir, classes=6,augmentation=get_validation_augmentation(),
                       preprocessing=get_preprocessing(preprocessing_fn))

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

save_dir = 'G:/MICCAI/log_seg/save'


for img,names in test_loader:
    id =names[0].split('\\')[-1]
    # print('id',id)
    image = img.to(DEVICE)
    pr_mask = best_model.predict(image)
    pr_mask = pr_mask.argmax(dim=1)
    import numpy as np
    pr_mask = (pr_mask.squeeze().cpu().numpy().round()).astype(np.float16)
    pro_mask = np.zeros(pr_mask.shape)*255
    pro_mask[pr_mask == 1] = 0
    pro_mask[pr_mask == 2] = 80
    pro_mask[pr_mask == 4] = 160
    save_file = os.path.join(save_dir,id)
    cv2.imwrite(save_file, pro_mask)
    print(id,'is done!')
