# 青光眼检测baseline原始代码

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
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101

import torchvision.transforms as trans

import warnings
warnings.filterwarnings('ignore')

# 配置
batchsize = 8 # 批大小,
image_size = 256
iters = 1000 # 迭代次数
val_ratio = 0.2 # 训练/验证数据划分比例，80 / 20
trainset_root = "../datasets/Train/Image"
val_root = "../datasets/Train/Image"
test_root = "../datasets/Validation/Image"
num_workers = 4
init_lr = 1e-6
optimizer_type = 'adam'

# 训练/验证数据集划分
filelists = os.listdir(trainset_root)
train_filelists, val_filelists = train_test_split(filelists, test_size=val_ratio, random_state=42)
print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))

# 数据加载（继承torch.utils.data中的Dataset类）
class GOALS_sub2_dataset(Dataset):
    def __init__(self,
                img_transforms,
                dataset_root,
                label_file='',
                filelists=None,
                numclasses=2,
                mode='train'):
        self.dataset_root = dataset_root
        self.img_transforms = img_transforms
        self.mode = mode.lower()
        self.num_classes = numclasses

        if self.mode == 'train':
            label = {row['ImgName']:row[1]
                    for _, row in pd.read_excel(label_file).iterrows()}
            self.file_list = [[f, label[int(f.split('.')[0])]] for f in os.listdir(dataset_root)]

        elif self.mode == "test":
            self.file_list = [[f, None] for f in os.listdir(dataset_root)]
        
        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]
    
    def __getitem__(self, idx):

        real_index, label = self.file_list[idx]
        img_path = os.path.join(self.dataset_root, real_index)    
        img = cv2.imread(img_path)
        
        if self.img_transforms is not None:
            img = self.img_transforms(img)
 
        # normlize on GPU to save CPU Memory and IO consuming.
        # img = (img / 255.).astype("float32")

        # img = img.transpose(2, 0, 1) # H, W, C -> C, H, W

        if self.mode == 'test':
            return img, real_index

        if self.mode == "train":            
            return img, label

    def __len__(self):
        return len(self.file_list)

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
    # trans.CropCenterSquare(),
    trans.Resize((image_size, image_size))
])

# 可视化并检查数据加载
_train = GOALS_sub2_dataset(dataset_root=trainset_root, 
                        img_transforms=img_train_transforms,
                        label_file='../datasets/Train/Train_GC_GT.xlsx')

plt.figure(figsize=(15, 5))

for i in range(5):
    img, lab = _train.__getitem__(i)
    plt.subplot(1, 5, i+1)
    plt.imshow(img.transpose(1, 2, 0))
    plt.axis("off")

_val = GOALS_sub2_dataset(dataset_root=trainset_root, 
                        img_transforms=img_val_transforms,
                        label_file='../datasets/Train/Train_GC_GT.xlsx')

plt.figure(figsize=(15, 5))
for i in range(5):
    img, lab = _val.__getitem__(i)
    plt.subplot(1, 5, i+1)
    plt.imshow(img.transpose(1, 2, 0))
    plt.axis("off")


# 网络模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.feature = resnet50(pretrained=True, num_classes=2) # 移除最后一层全连接
        # self.feature = resnet101(pretrained=True, num_classes=2) # 移除最后一层全连接
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, img):
        feature = self. feature(img)
        out1 = self.fc1(feature)
        logit = self.fc2(out1)

        return logit

# 功能函数
def train(model, iters, train_dataloader, val_dataloader, optimizer, criterion, log_interval, eval_interval):
    iter = 0
    model.train()
    avg_loss_list = []
    avg_acc_list = []
    best_acc = 0.
    while iter < iters:
        for data in train_dataloader:
            iter += 1
            if iter > iters:
                break
            imgs = (data[0] / 255.).astype("float32")
            labels = data[1].astype('int64')
            # print(labels)
            labels_ = paddle.unsqueeze(labels, axis=1)
            logits = model(imgs) 
            m = paddle.nn.Softmax()
            pred = m(logits)
            # print(pred.numpy())
            # print(pred.numpy().argmax(1))            
            acc = paddle.metric.accuracy(input=pred, label=labels_)
            one_hot_labels = paddle.fluid.layers.one_hot(labels_, 2, allow_out_of_range=False)
            loss = criterion(pred, one_hot_labels)            
            # print(loss.numpy())
            loss.backward()
            optimizer.step()

            model.clear_gradients()
            avg_loss_list.append(loss.numpy()[0])
            avg_acc_list.append(acc.numpy())
            

            if iter % log_interval == 0:
                avg_loss = np.array(avg_loss_list).mean()
                avg_acc = np.array(avg_acc_list).mean()
                avg_loss_list = []
                avg_acc_list = []
                print("[TRAIN] iter={}/{} avg_loss={:.4f} avg_acc={:.4f}".format(iter, iters, avg_loss, avg_acc))

            if iter % eval_interval == 0:
                avg_loss, avg_acc = val(model, val_dataloader, criterion)
                print("[EVAL] iter={}/{} avg_loss={:.4f} acc={:.4f}".format(iter, iters, avg_loss, avg_acc))
                if avg_acc >= best_acc:
                    best_acc = avg_acc
                    paddle.save(model.state_dict(),
                            os.path.join("/xx/task1_models/best_model_{:.4f}".format(best_acc), 'model.pdparams'))
                model.train()


def val(model, val_dataloader, criterion):
    model.eval()
    avg_loss_list = []
    avg_acc_list = []
    cache = []
    with paddle.no_grad():
        for data in val_dataloader:
            imgs = (data[0] / 255.).astype("float32")
            labels = data[1].astype('int64')
            labels_ = paddle.unsqueeze(labels, axis=1)
            logits = model(imgs)
            m = paddle.nn.Softmax()
            pred = m(logits)            
            acc = paddle.metric.accuracy(input=pred, label=labels_)
            one_hot_labels = paddle.fluid.layers.one_hot(labels_, 2, allow_out_of_range=False)
            loss = criterion(pred, one_hot_labels) 
            avg_loss_list.append(loss.numpy()[0])
            avg_acc_list.append(acc.numpy())        

    avg_loss = np.array(avg_loss_list).mean()
    acc = np.array(avg_acc_list).mean()

    return avg_loss, acc


# 训练阶段
img_train_transforms = trans.Compose([
    trans.RandomResizedCrop(
        image_size, scale=(0.90, 1.1), ratio=(0.90, 1.1)),
    trans.RandomHorizontalFlip(),
    trans.RandomVerticalFlip(),
    trans.RandomRotation(30)
])


img_val_transforms = trans.Compose([
    trans.CropCenterSquare(),
    trans.Resize((image_size, image_size))
])

train_dataset = GOALS_sub2_dataset(dataset_root=trainset_root, 
                        img_transforms=img_train_transforms,
                        filelists=train_filelists,
                        label_file='/xx/Train/Train_GC_GT.xlsx')

val_dataset = GOALS_sub2_dataset(dataset_root=trainset_root, 
                        img_transforms=img_val_transforms,
                        filelists=val_filelists,
                        label_file='/xx/Train/Train_GC_GT.xlsx')

train_loader = paddle.io.DataLoader(
    train_dataset,
    batch_sampler=paddle.io.DistributedBatchSampler(train_dataset, batch_size=batchsize, shuffle=True, drop_last=False),
    num_workers=num_workers,
    return_list=True,
    use_shared_memory=False
)

val_loader = paddle.io.DataLoader(
    val_dataset,
    batch_sampler=paddle.io.DistributedBatchSampler(val_dataset, batch_size=batchsize, shuffle=True, drop_last=False),
    num_workers=num_workers,
    return_list=True,
    use_shared_memory=False
)

model = Model()

if optimizer_type == "adam":
    optimizer = paddle.optimizer.Adam(init_lr, parameters=model.parameters())

criterion = nn.BCEWithLogitsLoss()
train(model, iters, train_loader, val_loader, optimizer, criterion, log_interval=10, eval_interval=100)


# 预测阶段
best_model_path = '/xx/task1_models/xx/model.pdparams'
model = Model()
para_state_dict = paddle.load(best_model_path)
model.set_state_dict(para_state_dict)
model.eval()

img_test_transforms = trans.Compose([
    trans.CropCenterSquare(),
    trans.Resize((image_size, image_size))
])

test_dataset = GOALS_sub2_dataset(dataset_root=test_root, 
                        img_transforms=img_test_transforms,
                        mode='test')
cache = []
for img, idx in test_dataset:
    img = img[np.newaxis, ...]
    img = paddle.to_tensor((img / 255.).astype("float32"))
    logits = model(img) 
    m = paddle.nn.Softmax()
    pred = m(logits)
    print(pred.numpy())
    cache.append([idx, pred.numpy()[0][1]])

submission_result = pd.DataFrame(cache, columns=['ImgName', 'GC_Pred'])
submission_result[['ImgName', 'GC_Pred']].to_csv("./submission_val_sub2.csv", index=False)