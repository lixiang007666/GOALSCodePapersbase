# task1训练代码，网络为unet和cenet，在celoss和diceloss的基础上使用不同的损失函数
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
from dataset_functions.task1_dataset import OCTDataset, OCTDataset_layermask
from dataset_functions.transforms import img_transforms, img_transforms_layermask
from models.task1_model import UNet
from models.cenet import CE_Net_, CE_Net_withSE, CE_Net_withECA, CE_Net_SEwoReduction, CE_Net_withECA_masklayer
import argparse
from loss import DiceLoss
from loss import FocalLoss
from loss import LabelSmoothingall, LabelSmoothingCrossEntropyLoss
import warnings
warnings.filterwarnings('ignore')

# 设置参数
# image_file = '../datasets/Train/Image'
# gt_file = '../datasets/Train/Layer_Masks'
orig_file = '../datasets/Train/Image'
image_file = '../datasets/Train/Image_aug2'  # aug2一张800*1100裁成两张800*800
gt_file = '../datasets/Train/Layer_Masks_aug2'
layer_file = '../datasets/Train/Layer_show_aug2'
image_size = 800 # 统一输入图像尺寸
val_ratio = 0.2 # 验证/训练图像划分比例
# batch_size = 4 # unet训练图像尺度800*800
batch_size = 8
iters = 2500
# init_lr = 1e-3  # unet
init_lr = 5e-4
optimizer_type = 'adam'
# os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')


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


def findlayer(gt_label): # gt_label 8*800*800
    # gt_label = gt_label.numpy() # batchsize*h*w
    gt_layer = torch.zeros(gt_label.size()[0], 5, gt_label.size()[2])
    for i in range(gt_label.size()[0]):
        for j in range(gt_label.size()[2]): # w
            for t in range(gt_label.size()[1]-1): # h
                if gt_label[i,t,j]==3 and gt_label[i,t+1,j]==0:
                    gt_layer[i,0,j]=t
                if gt_label[i,t,j]==0 and gt_label[i,t+1,j]==1:
                    gt_layer[i,1,j]=t
                if gt_label[i,t,j]==1 and gt_label[i,t+1,j]==2:
                    gt_layer[i,2,j]=t
                if gt_label[i,t,j]==2 and gt_label[i,t+1,j]==3:
                    gt_layer[i,3,j]=t
                if gt_label[i,t,j]==3 and gt_label[i,t+1,j]==0:
                    gt_layer[i,4,j]=t
    return gt_layer # batchsize*5*800




def train(model, iters, train_dataloader, val_dataloader, optimizer, criterion, metric, lsceloss, focalloss, l1loss, log_interval, evl_interval, device, fold):
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
            img, gt_label, gt_layer = data
            # gt_layer = findlayer(gt_label)
            # print(gt_label.shape)
            img, gt_label, gt_layer = img.to(device), gt_label.to(device), gt_layer.to(device)

            logits = model(img) # batchsize*9*800*800
            logits_mask = logits[:,0:4,:,:]
            logits_layer = logits[:,4:9,:,:] # batchsize*5*800*800
            logits_layer = F.softmax(logits_layer, dim=2) # batchsize*5*800*800
            col_num = torch.unsqueeze(torch.tensor([x for x in range(1,801)]), dim=1)
            col_num = col_num.repeat(gt_label.size()[0],5,1,800).to(device) # 800*1 --> 8*5*800*800
            predict_layer = torch.sum(torch.mul(logits_layer, col_num), dim=2).unsqueeze(dim=2) # 8*5*800
            gt_layer = torch.sum(torch.mul(gt_layer, col_num), dim=2).unsqueeze(dim=2)

            dice_loss, dice_score = metric(logits_mask, gt_label) # 预测mask的diceloss
            loss = criterion(logits_mask, gt_label)+dice_loss+l1loss(predict_layer, gt_layer)*0.1

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
                    torch.save(model.state_dict(), '/home/zhouziyu/miccai2022challenge/GOALS_code/checkpoints/task1/cenetECA_trainvalClearlySplit_masklayer_fold{}.pth'.format(fold))

                model.train()

# 验证函数
def val(model, val_dataloader, criterion, metric, device):
    model.eval()
    avg_loss_list = []
    avg_dice_list = []
    with torch.no_grad():
        for data in val_dataloader:
            img, gt_label, layer_label = data
            img, gt_label = img.to(device), gt_label.to(device)

            pred = model(img)
            dice_loss, dice_score = metric(pred[:,0:4,:,:], gt_label)
            loss = criterion(pred[:,0:4,:,:], gt_label)+dice_loss

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

img_train_transforms = img_transforms_layermask(applied_types='train')
img_val_transforms = img_transforms_layermask(applied_types='val')

train_dataset = OCTDataset_layermask(image_transforms=img_train_transforms, image_path=image_file, 
                                    layer_path=layer_file, filelists=train_filelists, gt_path=gt_file, mode='train')
val_dataset = OCTDataset_layermask(image_transforms=img_val_transforms, image_path=image_file, 
                                    layer_path=layer_file, filelists=val_filelists, gt_path=gt_file, mode='val')

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# model = UNet().to(device)
# model = CE_Net_().to(device)
# model = CE_Net_withSE().to(device)
# model = CE_Net_withECA().to(device)
model = CE_Net_withECA_masklayer().to(device)
# model = CE_Net_SEwoReduction().to(device)
# model = nn.DataParallel(model)

if optimizer_type == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr = init_lr)
criterion = torch.nn.CrossEntropyLoss().to(device)
metric = DiceLoss()
focalloss = FocalLoss(num_class=4)
# labelsmooth = LabelSmoothingall()
labelsmooth = LabelSmoothingCrossEntropyLoss()
l1loss = torch.nn.L1Loss()


# 开始训练
train(model, iters, train_loader, val_loader, optimizer, criterion, metric, labelsmooth, focalloss, l1loss, log_interval=10, evl_interval=50, device=device, fold=args.fold)
        