import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance, ImageFilter
import cv2

def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


#添加模糊
def Addblur(img,blur):
    if random.uniform(0, 1) < 0.5:  # 概率的判断
         #标准模糊
        if blur== "normal":
            img = img.filter(ImageFilter.BLUR)
            return img
        #高斯模糊
        if blur== "Gaussian":
            img = img.filter(ImageFilter.GaussianBlur)
            return img
        #均值模糊
        if blur== "mean":
            img = img.filter(ImageFilter.BoxBlur)
            return img

    else:
        return img


def randomCrop_new(image, label):

    image_width = image.size[0]
    # print(image_width)
    image_height = image.size[1]
    # print(image_height)

    a = np.random.randint(0, 5)
    if a==0:
        border = 30
        crop_win_width = np.random.randint(image_width - border, image_width)
        crop_win_height = np.random.randint(image_height - border, image_height)
    elif a==1:
        border = 60
        crop_win_width = np.random.randint(image_width - border-border, image_width-border)
        crop_win_height = np.random.randint(image_height - border-border, image_height-border)
    elif a==2:
        border = 90
        crop_win_width = np.random.randint(image_width - border-border, image_width-border)
        crop_win_height = np.random.randint(image_height - border-border, image_height-border)
    elif a==3:
        border = 120
        crop_win_width = np.random.randint(image_width - border-border, image_width-border)
        crop_win_height = np.random.randint(image_height - border-border, image_height-border)
    else:
        border = 150
        crop_win_width = np.random.randint(image_width - border-border, image_width-border)
        crop_win_height = np.random.randint(image_height - border-border, image_height-border)

    # print(crop_win_width,crop_win_height)
    #下取zheng
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    # print(random_region)
    return image.crop(random_region), label.crop(random_region)

def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)

def randomRotation(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, Image.NEAREST)
        label = label.rotate(random_angle, Image.NEAREST)
    return image, label

def randomRotation_90(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.5:
        a = np.random.randint(0, 3)
        if a == 0:
            image = image.rotate(90, Image.NEAREST)
            label = label.rotate(90, Image.NEAREST)
        if a == 1:
            image = image.rotate(180, Image.NEAREST)
            label = label.rotate(180, Image.NEAREST)
        if a == 2:
            image = image.rotate(-90, Image.NEAREST)
            label = label.rotate(-90, Image.NEAREST)

    return image, label
def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
# The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize_w,trainsize_h):
        self.trainsize_w = trainsize_w
        self.trainsize_h = trainsize_h
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.JPG') or f.endswith('.png') or f.endswith('.bmp')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png' )or f.endswith('.bmp')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize_w, self.trainsize_h)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            # transforms.Resize((self.trainsize_w, self.trainsize_h)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        # image = Addblur(image,'Gaussian')

        image, gt = cv_random_flip(image, gt)
        image, gt = randomCrop(image, gt)
        image, gt = randomRotation(image, gt)
        image = colorEnhance(image)

        image = self.img_transform(image)
        gt = gt.resize([self.trainsize_h, self.trainsize_w], Image.NEAREST)
        # gt = np.array(gt)
        gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        print(len(self.images),len(self.gts))
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize_h or w < self.trainsize_w:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, batchsize, trainsize_w,trainsize_h, shuffle=True, num_workers=12, pin_memory=False):
    dataset = SalObjDataset(image_root, gt_root, trainsize_w,trainsize_h)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

