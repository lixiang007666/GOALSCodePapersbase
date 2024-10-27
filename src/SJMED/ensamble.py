import cv2
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as trans
from models.task1_model import UNet
from models.cenet import CE_Net_, CE_Net_withSE, CE_Net_withECA, CE_Net_SEwoReduction, CE_Net_withECA_masklayer
from models.msnet import MSNet
from models.TransUNet.vit_seg_modeling import R50_ViT_B_16
from dataset_functions.task1_dataset import OCTDataset
from dataset_functions.transforms import img_transforms
from collections import OrderedDict
import segmentation_models_pytorch as smp



model0_path = './checkpoints/task1/cenetECA_labelsmooth_CEandDiceloss_fold0.pth'
model1_path = './checkpoints/task1/cenetECA_labelsmooth_CEandDiceloss_fold1.pth'
model2_path = './checkpoints/task1/cenetECA_labelsmooth_CEandDiceloss_fold2.pth'
model3_path = './checkpoints/task1/cenetECA_labelsmooth_CEandDiceloss_fold3.pth'
model4_path = './checkpoints/task1/cenetECA_labelsmooth_CEandDiceloss_fold4.pth'
model5_path = './checkpoints/task1/cenetECA_pseudolabel_labelsmooth3_0.05_focalCEandDiceloss_fold0.pth'
model6_path = './checkpoints/task1/cenetECA_pseudolabel_labelsmooth3_0.05_focalCEandDiceloss_fold1.pth'
model7_path = './checkpoints/task1/cenetECA_pseudolabel_labelsmooth3_0.05_focalCEandDiceloss_fold2.pth'
model8_path = './checkpoints/task1/cenetECA_pseudolabel_labelsmooth3_0.05_focalCEandDiceloss_fold3.pth'
model9_path = './checkpoints/task1/cenetECA_pseudolabel_labelsmooth3_0.05_focalCEandDiceloss_fold4.pth'
model10_path = './checkpoints/task1/cenetECA_pseudolabel_pickout_labelsmooth3_0.05_focalCEandDiceloss_fold0.pth'
model11_path = './checkpoints/task1/cenetECA_pseudolabel_pickout_labelsmooth3_0.05_focalCEandDiceloss_fold1.pth'
model12_path = './checkpoints/task1/cenetECA_pseudolabel_pickout_labelsmooth3_0.05_focalCEandDiceloss_fold2.pth'
model13_path = './checkpoints/task1/cenetECA_pseudolabel_pickout_labelsmooth3_0.05_focalCEandDiceloss_fold3.pth'
model14_path = './checkpoints/task1/cenetECA_pseudolabel_pickout_labelsmooth3_0.05_focalCEandDiceloss_fold4.pth'
model15_path = './checkpoints/task1/msnet_CEandDiceloss_fold0.pth'
model16_path = './checkpoints/task1/msnet_CEandDiceloss_fold1.pth'
model17_path = './checkpoints/task1/msnet_CEandDiceloss_fold2.pth'
model18_path = './checkpoints/task1/msnet_CEandDiceloss_fold3.pth'
model19_path = './checkpoints/task1/msnet_CEandDiceloss_fold4.pth'
model20_path = './checkpoints/task1/cenetECA_masklayer_fold0.pth'
model21_path = './checkpoints/task1/cenetECA_masklayer_fold1.pth'
model22_path = './checkpoints/task1/cenetECA_masklayer_fold2.pth'
model23_path = './checkpoints/task1/cenetECA_masklayer_fold3.pth'
model24_path = './checkpoints/task1/cenetECA_masklayer_fold4.pth'
model25_path = './checkpoints/task1/cenetECA_masklayer_minvall1loss_fold0.pth'
model26_path = './checkpoints/task1/cenetECA_masklayer_minvall1loss_fold1.pth'
model27_path = './checkpoints/task1/cenetECA_masklayer_minvall1loss_fold2.pth'
model28_path = './checkpoints/task1/cenetECA_masklayer_minvall1loss_fold3.pth'
model29_path = './checkpoints/task1/cenetECA_masklayer_minvall1loss_fold4.pth'

# model30_path = './checkpoints/task1/train_transunet_size800_batchsize8_aug4_CEandDiceloss_fold0.pth'
# model31_path = './checkpoints/task1/train_transunet_size800_batchsize8_aug4_CEandDiceloss_fold1.pth'
# model32_path = './checkpoints/task1/train_transunet_size800_batchsize8_aug4_CEandDiceloss_fold2.pth'
# model33_path = './checkpoints/task1/train_transunet_size800_batchsize8_aug4_CEandDiceloss_fold3.pth'
# model34_path = './checkpoints/task1/train_transunet_size800_batchsize8_aug4_CEandDiceloss_fold4.pth'
image_size = 800
test_root1 = '../datasets/Validation/Image_aug2_1'
test_root2 = '../datasets/Validation/Image_aug2_2'


model0 = CE_Net_withECA()
para_state_dict = torch.load(model0_path)
model0.load_state_dict(para_state_dict)
model0.eval()

model1 = CE_Net_withECA()
para_state_dict = torch.load(model1_path)
model1.load_state_dict(para_state_dict)
model1.eval()

model2 = CE_Net_withECA()
para_state_dict = torch.load(model2_path)
model2.load_state_dict(para_state_dict)
model2.eval()

model3 = CE_Net_withECA()
para_state_dict = torch.load(model3_path)
model3.load_state_dict(para_state_dict)
model3.eval()

model4 = CE_Net_withECA()
para_state_dict = torch.load(model4_path)
model4.load_state_dict(para_state_dict)
model4.eval()

model5 = CE_Net_withECA()
para_state_dict = torch.load(model5_path)
model5.load_state_dict(para_state_dict)
model5.eval()

model6 = CE_Net_withECA()
para_state_dict = torch.load(model6_path)
model6.load_state_dict(para_state_dict)
model6.eval()

model7 = CE_Net_withECA()
para_state_dict = torch.load(model7_path)
model7.load_state_dict(para_state_dict)
model7.eval()

model8 = CE_Net_withECA()
para_state_dict = torch.load(model8_path)
model8.load_state_dict(para_state_dict)
model8.eval()

model9 = CE_Net_withECA()
para_state_dict = torch.load(model9_path)
model9.load_state_dict(para_state_dict)
model9.eval()

model10 = CE_Net_withECA()
para_state_dict = torch.load(model10_path)
model10.load_state_dict(para_state_dict)
model10.eval()

model11 = CE_Net_withECA()
para_state_dict = torch.load(model11_path)
model11.load_state_dict(para_state_dict)
model11.eval()

model12 = CE_Net_withECA()
para_state_dict = torch.load(model12_path)
model12.load_state_dict(para_state_dict)
model12.eval()

model13 = CE_Net_withECA() 
para_state_dict = torch.load(model13_path)
model13.load_state_dict(para_state_dict)
model13.eval()

model14 = CE_Net_withECA()
para_state_dict = torch.load(model14_path)
model14.load_state_dict(para_state_dict)
model14.eval()

model15 = MSNet()
para_state_dict = torch.load(model15_path)
model15.load_state_dict(para_state_dict)
model15.eval()

model16 = MSNet()
para_state_dict = torch.load(model16_path)
model16.load_state_dict(para_state_dict)
model16.eval()

model17 = MSNet()
para_state_dict = torch.load(model17_path)
model17.load_state_dict(para_state_dict)
model17.eval()

model18 = MSNet()
para_state_dict = torch.load(model18_path)
model18.load_state_dict(para_state_dict)
model18.eval()

model19 = MSNet()
para_state_dict = torch.load(model19_path)
model19.load_state_dict(para_state_dict)
model19.eval()

model20 = CE_Net_withECA_masklayer()
para_state_dict = torch.load(model20_path)
model20.load_state_dict(para_state_dict)
model20.eval()

model21 = CE_Net_withECA_masklayer()
para_state_dict = torch.load(model21_path)
model21.load_state_dict(para_state_dict)
model21.eval()

model22 = CE_Net_withECA_masklayer()
para_state_dict = torch.load(model22_path)
model22.load_state_dict(para_state_dict)
model22.eval()

model23 = CE_Net_withECA_masklayer()
para_state_dict = torch.load(model23_path)
model23.load_state_dict(para_state_dict)
model23.eval()

model24 = CE_Net_withECA_masklayer()
para_state_dict = torch.load(model24_path)
model24.load_state_dict(para_state_dict)
model24.eval()

model25 = CE_Net_withECA_masklayer()
para_state_dict = torch.load(model25_path)
model25.load_state_dict(para_state_dict)
model25.eval()

model26 = CE_Net_withECA_masklayer()
para_state_dict = torch.load(model26_path)
model26.load_state_dict(para_state_dict)
model26.eval()

model27 = CE_Net_withECA_masklayer()
para_state_dict = torch.load(model27_path)
model27.load_state_dict(para_state_dict)
model27.eval()

model28 = CE_Net_withECA_masklayer()
para_state_dict = torch.load(model28_path)
model28.load_state_dict(para_state_dict)
model28.eval()

model29 = CE_Net_withECA_masklayer()
para_state_dict = torch.load(model29_path)
model29.load_state_dict(para_state_dict)
model29.eval()

# model30 = R50_ViT_B_16(in_ch=3,out_ch=4)
# para_state_dict = torch.load(model30_path)
# new_dict = OrderedDict()
# for k,v in para_state_dict.items():
#     name = k[7:]
#     new_dict[name] = v
# model30.load_state_dict(new_dict)
# model30.eval()

# model31 = R50_ViT_B_16(in_ch=3,out_ch=4)
# para_state_dict = torch.load(model31_path)
# new_dict = OrderedDict()
# for k,v in para_state_dict.items():
#     name = k[7:]
#     new_dict[name] = v
# model31.load_state_dict(new_dict)
# model31.eval()

# model32 = R50_ViT_B_16(in_ch=3,out_ch=4)
# para_state_dict = torch.load(model32_path)
# new_dict = OrderedDict()
# for k,v in para_state_dict.items():
#     name = k[7:]
#     new_dict[name] = v
# model32.load_state_dict(new_dict)
# model32.eval()

# model33 = R50_ViT_B_16(in_ch=3,out_ch=4)
# para_state_dict = torch.load(model33_path)
# new_dict = OrderedDict()
# for k,v in para_state_dict.items():
#     name = k[7:]
#     new_dict[name] = v
# model33.load_state_dict(new_dict)
# model33.eval()

# model34 = R50_ViT_B_16(in_ch=3,out_ch=4)
# para_state_dict = torch.load(model34_path)
# new_dict = OrderedDict()
# for k,v in para_state_dict.items():
#     name = k[7:]
#     new_dict[name] = v
# model34.load_state_dict(new_dict)
# model34.eval()

img_test_transforms = img_transforms(applied_types='test')

test_dataset1 = OCTDataset(image_transforms=img_test_transforms,
                        image_path=test_root1,
                        mode='test')

test_dataset2 = OCTDataset(image_transforms=img_test_transforms,
                        image_path=test_root2,
                        mode='test')


cache = []
for img1, idx1, h, w in test_dataset1:

    for img2, idx2, h, w in test_dataset2:
        if idx2 == idx1:
            img1 = img1.unsqueeze(0) # 增加维度：(3,800,800)-->(1,3,800,800)
            img2 = img2.unsqueeze(0)
            logits1 = 0
            logits2 = 0
            for i in range(20):
                logits1 += locals()['model'+str(i)](img1)
                logits2 += locals()['model'+str(i)](img2)
            for i in range(20,30):
                logits1 += locals()['model'+str(i)](img1)[:,0:4,:,:]
                logits2 += locals()['model'+str(i)](img2)[:,0:4,:,:]
            logits1 = logits1/30
            logits2 = logits2/30
            # logits1 = (model0(img1)+model1(img1)+model2(img1)+model3(img1)+model4(img1))/5 # 模型输出维度(1,4,800,800)
            # logits2 = (model0(img2)+model1(img2)+model2(img2)+model3(img2)+model4(img2))/5

            m = torch.nn.Softmax(dim=1)
            logits1 = m(logits1).detach().numpy()
            logits2 = m(logits2).detach().numpy()

            overlap1 = logits1[:,:,:,300:800]
            overlap2 = logits2[:,:,:,0:500]
            overlap = (overlap1+overlap2)/2

            logits = np.concatenate((logits1[:,:,:,0:300], overlap, logits2[:,:,:,500:800]), axis=3)


            pred_img = logits.argmax(1)
            pred_gray = np.squeeze(pred_img, axis=0)
            pred_gray = pred_gray.astype('float32')

            pred_gray[pred_gray == 1] = 80
            pred_gray[pred_gray == 2] = 160
            pred_gray[pred_gray == 3] = 255
            # pred = cv2.resize(pred_gray, (w, h))
            # print(pred_gray.shape)
            print(idx1)
            cv2.imwrite('./submission/task1/ensamble3_ceneteca_masnet_cenetecamasklayer/'+idx1, pred_gray)
