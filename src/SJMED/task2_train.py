# 青光眼检测baseline

# 导入库
import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import auc
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from models.task2_model import Model
from dataset_functions.task2_dataset import GOALS_sub2_dataset
import torchvision.transforms as trans

import warnings
warnings.filterwarnings('ignore')

# 配置
batchsize = 8 # 批大小,
image_size = 256
iters = 1000 # 迭代次数
val_ratio = 0.2 # 训练/验证数据划分比例，80 / 20
trainset_root = '../datasets/Train/Image'
val_root = '../datasets/Train/Image'
train_label_root = '../datasets/Train/Train_GC_GT.xlsx'
test_root = '../datasets/Validation/Image'
num_workers = 4
init_lr = 1e-6
optimizer_type = 'adam'
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# 训练/验证数据集划分
filelists = os.listdir(trainset_root) # 返回文件夹中包含文件名字的列表
train_filelists, val_filelists = train_test_split(filelists, test_size=val_ratio, random_state=42)
print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))
print(val_filelists)


# plot loss and accuracy
def plot_loss(loss):
    plt.figure()
    # loss = np.array(torch.tensor(loss, device='cpu'))
    plt.plot(loss, 'b-', label = 'val loss')
    plt.legend(fontsize=10)
    plt.title('loss', fontsize = 10)
    plt.xlabel('Iteration/100')
    plt.savefig('./figures/task2/loss.png')

def plot_accuracy(acc):
    plt.figure()
    # acc = np.array(acc)
    plt.plot(acc, 'b-', label = 'val acc')
    plt.legend(fontsize=10)
    plt.title('accuracy', fontsize = 10)
    plt.xlabel('Iteration/100')
    plt.savefig('./figures/task2/accuracy.png')


# 功能函数
def train(model, iters, train_dataloader, val_dataloader, optimizer, criterion, log_interval, eval_interval, device):
    iter = 0
    model.train()
    avg_loss_list = []
    avg_acc_list = []
    best_acc = 0.
    n_correct = 0
    n_total = 0
    while iter < iters:
        for _, data in enumerate(train_dataloader): # 便利dataloader得到的data为一个batch里的图像和label
            iter += 1
            # print(iter)
            if iter > iters:
                break
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            # imgs = (data[0] / 255.).to(torch.float32).to(device)
            # labels = data[1].to(torch.int64).to(device)
            # print(labels)
            # labels_ = torch.unsqueeze(labels, axis=1) # 维度扩展：一维80-->二维80*1
            logits = model(imgs) 
            # m = paddle.nn.Softmax()
            # pred = m(logits)
            # print(pred.numpy())
            # print(pred.numpy().argmax(1))            
            # acc = paddle.metric.accuracy(input=pred, label=labels_)
            _, indices = logits.max(dim=1) # 找出行最大值，返回索引
            n_correct += sum(indices==labels)
            n_total += len(labels)
            acc = n_correct.cpu().detach().numpy() * 1.0 /n_total
            # one_hot_labels = paddle.fluid.layers.one_hot(labels_, 2, allow_out_of_range=False)
            loss = criterion(logits, labels)            
            # print(loss.numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            # model.clear_gradients()
            avg_loss_list.append(loss.cpu().detach().numpy())
            avg_acc_list.append(acc)
            

            if iter % log_interval == 0:
                avg_loss = np.array(avg_loss_list).mean()
                avg_acc = np.array(avg_acc_list).mean()
                avg_loss_list = [] # 存log_interval个iter的loss和acc
                avg_acc_list = []
                print('[TRAIN] iter={}/{} avg_loss={:.4f} avg_acc={:.4f}'.format(iter, iters, avg_loss, avg_acc))

            if iter % eval_interval == 0:
                avg_loss, avg_acc = val(model, val_dataloader, criterion, device)
                print('[EVAL] iter={}/{} avg_loss={:.4f} acc={:.4f}'.format(iter, iters, avg_loss, avg_acc))
                if avg_acc >= best_acc:
                    best_acc = avg_acc
                    torch.save(model.state_dict(), '/home/zhouziyu/miccai2022challenge/GOALS_code/checkpoints/task2/train1.pth')
                model.train()

def val(model, val_dataloader, criterion, device):
    model.eval()
    loss_list = []
    acc_list = []
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for _, data in enumerate(val_dataloader):
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            # imgs = data[0]
            # labels = data[1]
            logits = model(imgs)
            _, indices = logits.max(dim=1) # 找出行最大值，返回索引
            n_correct += sum(indices==labels)
            n_total += len(labels)
            acc = n_correct.cpu().numpy() * 1.0 /n_total
            loss = criterion(logits, labels)     
            loss_list.append(loss.cpu().detach().numpy())
            acc_list.append(acc)

    avg_loss = np.array(loss_list).mean()
    avg_acc = np.array(acc_list).mean()
    plot_loss(loss_list)
    plot_accuracy(acc_list)

    return avg_loss, avg_acc      


# 训练阶段

# 数据增强
img_train_transforms = trans.Compose([
    trans.ToTensor(),
    trans.RandomResizedCrop(
        image_size, scale=(0.90, 1.1), ratio=(0.90, 1.1)), # scale与原图像面积大小范围，ratio剪裁后宽高比
    trans.RandomHorizontalFlip(), # 随机水平翻转
    trans.RandomVerticalFlip(), # 随机垂直翻转
    trans.RandomRotation(30)
])

img_val_transforms = trans.Compose([
    trans.ToTensor(),
    # trans.CenterCrop(image_size),
    trans.Resize((image_size, image_size))
])

train_dataset = GOALS_sub2_dataset(img_transforms=img_train_transforms,
                                    dataset_root=trainset_root,
                                    filelists=train_filelists,
                                    label_file=train_label_root,
                                    mode='train'
                                    )

val_dataset = GOALS_sub2_dataset(img_transforms=img_val_transforms,
                                dataset_root=trainset_root,
                                filelists=val_filelists,
                                label_file=train_label_root,
                                mode='val'
                                )

train_loader = DataLoader(dataset=train_dataset,
                        batch_size=batchsize,
                        num_workers=0,
                        shuffle=True,
                        drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=batchsize,
                        num_workers=0,
                        shuffle=True,
                        drop_last=True)


model = Model().to(device)

if optimizer_type == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr = init_lr)
criterion = torch.nn.CrossEntropyLoss().to(device)

train(model, iters, train_loader, val_loader, optimizer, criterion, log_interval=10, eval_interval=100, device=device)


# # 预测阶段
# best_model_path = '/xx/task1_models/xx/model.pdparams'
# model = Model()
# para_state_dict = paddle.load(best_model_path)
# model.set_state_dict(para_state_dict)
# model.eval()

# img_test_transforms = trans.Compose([
#     trans.CropCenterSquare(),
#     trans.Resize((image_size, image_size))
# ])

# test_dataset = GOALS_sub2_dataset(dataset_root=test_root, 
#                         img_transforms=img_test_transforms,
#                         mode='test')
# cache = []
# for img, idx in test_dataset:
#     img = img[np.newaxis, ...]
#     img = paddle.to_tensor((img / 255.).astype("float32"))
#     logits = model(img) 
#     m = paddle.nn.Softmax()
#     pred = m(logits)
#     print(pred.numpy())
#     cache.append([idx, pred.numpy()[0][1]])

# submission_result = pd.DataFrame(cache, columns=['ImgName', 'GC_Pred'])
# submission_result[['ImgName', 'GC_Pred']].to_csv("./submission_val_sub2.csv", index=False)

# class GOALS_sub2_dataset(Dataset):
#     def __init__(self,
#                 img_transforms,
#                 dataset_root,
#                 label_file='',
#                 filelists=None,
#                 numclasses=2,
#                 mode='train'):
#                 self.img_transforms = img_transforms
#                 self.dataset_root = dataset_root
#                 self.mode = mode.lower()
#                 self.num_classes = numclasses

#                 if self.mode == 'train': # 如果输入了mode，train或test，数据集为dataset_root中图片
#                     label = {row['ImgName']:row[1]
#                             for _, row in pd.read_excel(label_file).iterrows()}
#                     self.file_list = [[f, label[int(f.split('.')[0])]] for f in os.listdir(dataset_root)]

#                 elif self.mode == "test": # 测试集没有label
#                     self.file_list = [[f, None] for f in os.listdir(dataset_root)]
                
#                 if filelists is not None:  # 如果输入filelists，数据集为filelists中图片,dataset_root不起作用
#                     self.file_list = [item for item in self.file_list if item[0] in filelists]
    
#     def __getitem__(self, idx):

#         real_index, label = self.file_list[idx]
#         # real_index = np.array(real_index)
#         # real_index = torch.from_numpy(real_index)
#         label = np.array(label)
#         label = torch.from_numpy(label)
#         img_path = os.path.join(self.dataset_root, real_index)    
#         img = cv2.imread(img_path)
#         img = trans.ToTensor(img)
        
#         if self.img_transforms is not None:
#             img = self.img_transforms(img)
#             img = img/255
 
#         # normlize on GPU to save CPU Memory and IO consuming.
#         # img = (img / 255.).astype("float32")
#         # print(img.shape)
#         # img = img.transpose(2, 0).transpose(1,2) # H, W, C -> C, H, W

#         if self.mode == 'test':
#             return img.float(), real_index

#         if self.mode == "train":            
#             return img.float(), label.long()

#     def __len__(self):
#         return len(self.file_list)