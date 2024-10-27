import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np


def oct_data_aug(imgs, masks, size):
    # 标准化格式
    imgs = np.array(imgs)
    masks = np.array(masks).astype(np.uint8)

    # 注意mask新填充的像素不能为黑色0（对应了目标的label），应置为白色255（背景类）
    masks[masks == 0] = 25
    masks[masks == 255] = 0
    masks[masks == 25] = 255

    seq = iaa.Sequential(
        [
            # iaa.CropToFixedSize(width=size, height=size),

            iaa.Fliplr(0.5),  # 50%图像进行水平翻转
            iaa.Flipud(0.5),  # 50%图像做垂直翻转

            # 分段仿射
            iaa.PiecewiseAffine(scale=(0.01, 0.05)),

            # 伽马变换
            iaa.GammaContrast((0.5, 2.0)),

            # 随机弹性变换
            # iaa.ElasticTransformation(alpha=300, sigma=30),
        ],
        random_order=True  # 随机的顺序把这些操作用在图像上
    )

    seq_det = seq.to_deterministic()  # 确定一个数据增强的序列
    # print('imgs.shape',imgs.shape)
    segmaps = ia.SegmentationMapsOnImage(masks, shape=masks.shape)  # 分割标签格式
    image_aug, segmaps_aug_iaa = seq_det(image=imgs, segmentation_maps=segmaps)  # 将方法同步应用在图像和分割标签上，
    segmap_aug = segmaps_aug_iaa.get_arr().astype(np.uint8)  # 转换成np类型

    segmap_aug[segmap_aug == 255] = 25
    segmap_aug[segmap_aug == 0] = 255
    segmap_aug[segmap_aug == 25] = 0

    return image_aug, segmap_aug

