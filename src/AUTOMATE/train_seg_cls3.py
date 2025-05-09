import warnings
warnings.filterwarnings('ignore')

import paddle
import paddleseg
from paddleseg import transforms as T
from paddleseg.core import train
import cv2
import numpy as np

train_transforms = [
    T.ResizeStepScaling(0.75, 1.25, 0),
    T.RandomRotation(max_rotation=30),
    T.RandomHorizontalFlip(prob=0.5),
    T.RandomDistort(
        brightness_range=0.2, brightness_prob=0.5,
        contrast_range=0.4, contrast_prob=0.5,
        saturation_range=0.4, saturation_prob=0.5,
        hue_range=0, hue_prob=0,
        sharpness_range=0, sharpness_prob=0),
    T.RandomBlur(prob=0.1, blur_type='random'),
    T.RandomPaddingCrop(crop_size=(1100, 800)),
    T.Normalize(mean=[0.22894311, 0.22894311, 0.22894311], std=[0.16314624, 0.16314624, 0.16314624])
]

eval_transforms = [
    T.Normalize(mean=[0.22894311, 0.22894311, 0.22894311], std=[0.16314624, 0.16314624, 0.16314624])
]

train_dataset = paddleseg.datasets.Dataset(
    mode='train',
    num_classes=2,
    dataset_root='GOALS2022-Train_cls3/Train',
    train_path='split_lists/seg_holdout/train.txt',
    transforms=train_transforms,
    edge=False)

eval_dataset = paddleseg.datasets.Dataset(
    mode='val',
    num_classes=2,
    dataset_root='GOALS2022-Train_cls3/Train',
    val_path='split_lists/seg_holdout/val.txt',
    transforms=eval_transforms)

model = paddleseg.models.FCN(
    num_classes=2,
    backbone=paddleseg.models.backbones.HRNet_W18(),
    backbone_indices=(-1,),
    pretrained='https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_hrnetw18_cityscapes_1024x512_80k/model.pdparams')

iters = 5000
train_batch_size = 2
learning_rate = 0.0001

decayed_lr = paddle.optimizer.lr.CosineAnnealingDecay(
    learning_rate=learning_rate,
    T_max=iters)

decayed_lr = paddle.optimizer.lr.LinearWarmup(
    learning_rate=decayed_lr,
    warmup_steps=250,
    start_lr=0.0,
    end_lr=learning_rate)

optimizer = paddle.optimizer.AdamW(
    learning_rate=decayed_lr,
    parameters=model.parameters())

from paddleseg.models import DiceLoss

losses = {
    'types': [DiceLoss(weight=[0.95, 1.05])],
    'coef': [1]
}

train(
    train_dataset=train_dataset,
    val_dataset=eval_dataset,

    model=model,
    optimizer=optimizer,
    losses=losses,

    iters=iters,
    batch_size=train_batch_size,

    save_interval=200,
    log_iters=10,
    save_dir='models_seg/cls3_FCN18_WDice',
    use_vdl=True,
    keep_checkpoint_max=1)
