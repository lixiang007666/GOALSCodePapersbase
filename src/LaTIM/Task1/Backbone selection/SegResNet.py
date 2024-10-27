import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

from sklearn.model_selection import train_test_split
import monai
from PIL import Image
from monai.losses.dice import DiceLoss
from torchvision.transforms.functional import to_pil_image,affine

### 设置参数
images_file = '../GOALS2022-Train/Train/Image'  # 训练图像路径
gt_file = '../GOALS2022-Train/Train/Layer_Masks'
image_size = 800 # 输入图像统一尺寸
val_ratio = 0.15  # 训练/验证图像划分比例
batch_size = 4 # 批大小
iters = 10000 # 训练迭代次数
optimizer_type = 'adam' # 优化器, 可自行使用其他优化器，如SGD, RMSprop,...
num_workers = 8 # 数据加载处理器个数
init_lr = 1e-3 # 初始学习率


summary_dir = './logs'
torch.backends.cudnn.benchmark = True
print('cuda',torch.cuda.is_available())
print('gpu number',torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))
summaryWriter = SummaryWriter(summary_dir)

# 训练/验证数据集划分
filelists = os.listdir(images_file)
print(filelists)
train_filelists, val_filelists = train_test_split(filelists, test_size = val_ratio)
print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))


### 从数据文件夹中加载眼底图像，提取相应的金标准，生成训练样本
class OCTDataset(Dataset):
    def __init__(self, image_file, gt_path=None, filelists=None,  mode='train'):
        super(OCTDataset, self).__init__()
        self.mode = mode
        self.image_path = image_file
        image_idxs = os.listdir(self.image_path) # 0001.png,
        self.gt_path = gt_path
        self.file_list = [image_idxs[i] for i in range(len(image_idxs))]        
        if filelists is not None:
            self.file_list = [item for item in self.file_list if item in filelists] 
    
    def transform(self,img, mask):
        (d, t, sc, sh) = transforms.RandomAffine.get_params(degrees=(-20, 20), translate=(0.2, 0.2),
                                                            scale_ranges=(0.8, 1.2), shears=(-20, 20),
                                                            img_size=img.shape)
        img = affine(to_pil_image(img), angle=d, translate=t, scale=sc, shear=sh)
        mask = affine(to_pil_image(mask), angle=d, translate=t, scale=sc, shear=sh)

        return (np.array(img), np.array(mask))
   
    def __getitem__(self, idx):
        real_index = self.file_list[idx]
        img_path = os.path.join(self.image_path, real_index)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE) 
        #img = img[:,:,np.newaxis]
        #print(img.shape)
        h,w = img.shape # (800, 1100, 3)     
        img = cv2.resize(img,(image_size, image_size))
        #img = img[:,:,np.newaxis]
        #print(img.shape)
        
        if self.mode == 'train' or self.mode == 'val':
            gt_tmp_path = os.path.join(self.gt_path, real_index)
            gt_img = cv2.imread(gt_tmp_path)

            ### 像素值为0的是RNFL(类别 0)，像素值为80的是GCIPL(类别 1)，像素值为160的是脉络膜(类别 2)，像素值为255的是其他（类别3）。
            gt_img[gt_img == 0] = 3
            gt_img[gt_img == 80] = 1
            gt_img[gt_img == 160] = 2
            gt_img[gt_img == 255] = 0
            
            gt_img = cv2.resize(gt_img,(image_size, image_size),interpolation = cv2.INTER_NEAREST)
            gt_img = gt_img[:,:,1]
            #print('gt shape', gt_img.shape)           
        
        if self.mode == 'train':
            img, gt_img = self.transform(img, gt_img)
        
        if self.mode == 'train' or self.mode == 'val':
            gt_img = gt_img[:,:,np.newaxis]
            gt_img = gt_img.transpose(2,0,1)
            gt_img = torch.from_numpy(gt_img)
            
        #print(img.shape)
        img = img[:,:,np.newaxis]
        img = img.transpose(2, 0, 1) # H, W, C -> C, H, W
        img = torch.from_numpy(img)
        
        
        # print(img.shape)
        # img = img_re.astype(np.float32)
        
        
        if self.mode == 'test':
            ### 在测试过程中，加载数据返回眼底图像，数据名称，原始图像的高度和宽度
            return img, real_index, h, w
        
        if self.mode == 'train' or self.mode == 'val':
            ###在训练过程中，加载数据返回眼底图像及其相应的金标准           
            return img, gt_img

    def __len__(self):
        return len(self.file_list)


model = monai.networks.nets.SegResNet(in_channels=1, out_channels=4,spatial_dims=2)
        
        

model.cuda()

metric = DiceLoss(to_onehot_y = True, softmax = True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = ExponentialLR(optimizer, gamma=0.99)

train_dataset = OCTDataset(image_file = images_file, 
                        gt_path = gt_file,
                        filelists=train_filelists)

val_dataset = OCTDataset(image_file = images_file, 
                        gt_path = gt_file,
                        filelists=val_filelists,mode='val')


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                        pin_memory=True)


num_epochs = 1000
for epoch in range(num_epochs):
    #print('lr now = ', get_learning_rate(optimizer))
    avg_loss_list = []
    avg_dice_list = []
    
    model.train()
    with torch.enable_grad():
        for batch_idx, data in enumerate(train_loader):
            img = (data[0]).float()
            gt_label = (data[1])
            #print(img.shape)
            #print(gt_label.shape)
            
            img = img.cuda()
            gt_label = gt_label.cuda()
            
            
            logits = model(img)
            #print(logits)
            dice = metric(logits,gt_label)
            loss = criterion(logits, torch.squeeze(gt_label,dim=1).long()) + dice
            #print(loss)
            
            avg_loss_list.append(loss.item())
            avg_dice_list.append(dice.item())
            

            loss.backward()
            optimizer.step()
            for param in model.parameters():
                param.grad = None
            
        avg_loss = np.array(avg_loss_list).mean()
        avg_dice = np.array(avg_dice_list).mean()
        print("[TRAIN] epoch={}/{} avg_loss={:.4f} avg_dice={:.4f}".format(epoch, num_epochs, avg_loss, avg_dice))
        summaryWriter.add_scalars('loss', {"loss": (avg_loss)}, epoch)
        summaryWriter.add_scalars('dice', {"dice": avg_dice}, epoch)
        
    model.eval()
    critere = monai.metrics.SurfaceDistanceMetric(include_background=True,reduction="mean")
    val_avg_list = []
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            
            img = (data[0]).float()
            gt_label = (data[1])

            img = img.cuda()
            gt_label = gt_label.cuda()


            logits = model(img)
            pred_img = logits.argmax(1)
            #print(pred_img.shape)
            #print(gt_label.shape)
            
            out = monai.networks.utils.one_hot(pred_img.view(1,1,image_size,image_size), num_classes=4, dim=1)
            #print(out.shape)
            oh_gt = monai.networks.utils.one_hot(gt_label, num_classes=4, dim=1)
            score = critere(out,oh_gt)
            critere.reset()
            
            
            #dice = metric(logits,gt_label)
            val_avg_list.append(torch.mean(score).item())
        
        val_distance = np.array(val_avg_list).mean()
        print("[EVAL] epoch={}/{}  val_distance={:.4f}".format(epoch, num_epochs,val_distance))
        summaryWriter.add_scalars('val_distance', {"val_distance": val_distance}, epoch)
        
    scheduler.step()

    filepath = './weights'
    folder = os.path.exists(filepath)
    if not folder:
        # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(filepath)
    path = './weights/model' + str(epoch) + '_'+ str(val_distance) + '.pth'
    torch.save(model.state_dict(), path)            

summaryWriter.close()




