# task1训练代码，网络为transunet
from cProfile import label
import os
from statistics import mode
from tkinter import N
from turtle import forward
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as trans
from dataset_functions.task1_dataset import OCTDataset
from dataset_functions.transforms import img_transforms, img_transforms_more
from models.task1_model import UNet
from models.cenet import CE_Net_
from models.TransUNet.vit_seg_modeling import R50_ViT_B_16
import argparse
from loss import FocalLoss

# 设置参数
# image_file = '../datasets/Train/Image'
# gt_file = '../datasets/Train/Layer_Masks'
orig_file = '../datasets/Train/Image_Pseudolabel'
image_file = '../datasets/Train/Image_Pseudolabel_aug2'  # 使用剪裁后的图片
gt_file = '../datasets/Train/Image_Pseudolabel_aug2'
image_size = 800 # 统一输入图像尺寸
val_ratio = 0.2 # 验证/训练图像划分比例
# batch_size = 4 # unet训练图像尺度800*800
batch_size = 8
iters = 2500
# init_lr = 1e-3  # unet
init_lr = 5e-4
optimizer_type = 'adam'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# 训练集、验证集划分
filelists = os.listdir(image_file)
orig_filelists = os.listdir(orig_file)
# train_filelists, val_filelists = train_test_split(filelists, test_size=val_ratio, random_state=42)
# print('Total Nums: {}, train: {}, val: {}'.format(len(filelists), len(train_filelists), len(val_filelists)))

# 五折交叉验证，生成五套训练集、测试集：train_filelists0 ~ train_filelists4, test_filelists0 ~ test_flelists4
# rkf = KFold(n_splits=5)
# for fold, (train_index, test_index) in enumerate(rkf.split(filelists)):
#     locals()['train_index'+str(fold)] = train_index # locals()来动态定义变量
#     locals()['test_index'+str(fold)] = test_index
#     locals()['train_filelists'+str(fold)] = []
#     locals()['test_filelists'+str(fold)] = []
#     for i in locals()['train_index'+str(fold)]:
#         locals()['train_filelists'+str(fold)].append(filelists[i])
#     for i in locals()['test_index'+str(fold)]:
#         locals()['test_filelists'+str(fold)].append(filelists[i])
rkf = KFold(n_splits=5)
for fold, (train_index, test_index) in enumerate(rkf.split(orig_filelists)):
    locals()['train_index'+str(fold)] = train_index # locals()来动态定义变量
    locals()['test_index'+str(fold)] = test_index
    locals()['train_filelists'+str(fold)] = []
    locals()['test_filelists'+str(fold)] = []
    for i in locals()['train_index'+str(fold)]:
        img_name = orig_filelists[i]
        img_idx = img_name.split('.')[0]
        locals()['train_filelists'+str(fold)].append(img_idx+'_1.png')
        locals()['train_filelists'+str(fold)].append(img_idx+'_2.png')

    for i in locals()['test_index'+str(fold)]:
        img_name = orig_filelists[i]
        img_idx = img_name.split('.')[0]
        locals()['test_filelists'+str(fold)].append(img_idx+'_1.png')
        locals()['test_filelists'+str(fold)].append(img_idx+'_2.png')


class DiceLoss(nn.Module):
    """
    Implements the dice loss function.
    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """
    def __init__(self, ignore_index = 3): # 无用像素是其他类别，mask像素值设为3
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.eps = 1e-5 # 防止分母为0加上此参数

    def forward(self, logits, labels):
        if len(labels.shape) != len(logits.shape):
            labels = torch.unsqueeze(labels, 1) # labels维度batchsize*1*h*w
        num_classes = logits.shape[1] # 分割类别
        # mask = (labels != self.ignore_index)
        # mask = labels
        # logits = logits * mask
        single_label_list = []

        for c in range(num_classes):
            single_label = (labels == c)
            single_label = torch.squeeze(single_label, 1)
            single_label_list.append(single_label)
        labels_one_hot = torch.stack(tuple(single_label_list), axis = 1) # 将label转换为oen-hot，维度：batchsize*numclasses*h*w，使用tuple是因为其不可变，代替list更安全
        logits = F.softmax(logits, dim = 1) # logits维度batchsize*4*h*w
        dims = (0,2,3) # 压缩0，2，3这三个维度，最后得到的loss是一个长度为4的一维向量，其值分别为4个类别的dice
        intersection = torch.sum(logits * labels_one_hot, dims)
        cardinality = torch.sum(logits + labels_one_hot, dims)
        dice_score = (2. * intersection / (cardinality + self.eps))
        dice_loss = (1-dice_score).mean()
        return dice_loss, dice_score # 返回四个类别的dice_score

def train(model, iters, train_dataloader, val_dataloader, optimizer, criterion, metric, focalloss, log_interval, evl_interval, device, fold):
    iter = 0
    model.train()
    avg_loss_list = []
    avg_dice_list = []
    best_dice = 0
    while iter < iters:
        for _, data in enumerate(train_dataloader):
            iter += 1
            if iter > iters:
                break
            img, gt_label = data
            # print(gt_label.shape)
            img = img.to(device)
            gt_label = gt_label.to(device)

            logits = model(img)
            dice_loss, dice_score = metric(logits, gt_label)
            loss = criterion(logits, gt_label)+dice_loss+focalloss(logits,gt_label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            avg_loss_list.append(loss.cpu().detach().numpy())
            avg_dice_list.append(dice_score.mean().cpu().detach().numpy())

            if iter % log_interval == 0:
                avg_loss = np.array(avg_loss_list).mean()
                avg_dice = np.array(avg_dice_list).mean()
                avg_loss_list = []
                avg_dice_list = []
                print("fold{}: [TRAIN] iter={}/{} avg_loss={:.4f} dice={}".format(fold, iter, iters, avg_loss, dice_score))

            if iter % evl_interval == 0:
                avg_loss, avg_dice, val_dice_score = val(model, val_dataloader, criterion, metric, device)
                print("[EVAL] iter={}/{} avg_loss={:.4f} val_dice_score={}".format(iter, iters, avg_loss, val_dice_score))
                if avg_dice >= best_dice:
                    best_dice = avg_dice
                    torch.save(model.state_dict(), '/home/zhouziyu/miccai2022challenge/GOALS_code/checkpoints/task1/train_transunet_size800_batchsize8_aug4_CEandDiceloss_fold{}.pth'.format(fold))

                model.train()

# 验证函数
def val(model, val_dataloader, criterion, metric, device):
    model.eval()
    avg_loss_list = []
    avg_dice_list = []
    with torch.no_grad():
        for data in val_dataloader:
            img, gt_label = data
            img = img.to(device)
            gt_label = gt_label.to(device)

            pred = model(img)
            dice_loss, dice_score = metric(pred, gt_label)
            loss = criterion(pred, gt_label)+dice_loss

            avg_loss_list.append(loss.cpu().detach().numpy())
            avg_dice_list.append(dice_score.cpu().detach().numpy())

    avg_loss = np.array(avg_loss_list).mean() # list转array
    avg_dice = np.array(avg_dice_list).mean()

    return avg_loss, avg_dice, dice_score

# 训练阶段

# 设置传入参数
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fold', help='use k-fold train and test data')
args = parser.parse_args()

train_filelists = locals()['train_filelists'+str(args.fold)] 
val_filelists = locals()['test_filelists'+str(args.fold)] 


# 生成训练集和验证集
# img_train_transforms = trans.Compose([
#                                     trans.ToTensor(), 
#                                     ])
# img_val_transforms = trans.Compose([
#                                     trans.ToTensor(), 
#                                     ])

# img_train_transforms = img_transforms(applied_types='train')
# img_val_transforms = img_transforms(applied_types='val')

img_train_transforms = img_transforms_more(applied_types='train')
img_val_transforms = img_transforms_more(applied_types='val')

train_dataset = OCTDataset(image_transforms=img_train_transforms, image_path=image_file, filelists=train_filelists, gt_path=gt_file, mode='train')
val_dataset = OCTDataset(image_transforms=img_val_transforms, image_path=image_file, filelists=val_filelists, gt_path=gt_file, mode='val')

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# model = UNet().to(device)
model = R50_ViT_B_16(in_ch=3,out_ch=4).to(device)
model = nn.DataParallel(model)

if optimizer_type == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr = init_lr)
criterion = torch.nn.CrossEntropyLoss().to(device)
metric = DiceLoss()
focalloss = FocalLoss(num_class=4)


# 开始训练
train(model, iters, train_loader, val_loader, optimizer, criterion, metric, focalloss, log_interval=10, evl_interval=50, device=device, fold=args.fold)
        