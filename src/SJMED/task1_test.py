import cv2
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as trans
from models.task1_model import UNet
from models.cenet import CE_Net_, CE_Net_withSE, CE_Net_withECA, CE_Net_SEwoReduction
from models.TransUNet.vit_seg_modeling import R50_ViT_B_16
from dataset_functions.task1_dataset import OCTDataset
from dataset_functions.transforms import img_transforms
from models.msnet import MSNet, LossNet



# # 五折交叉验证
# # 预测阶段
# model0_path = './checkpoints/task1/train_size800_batchsize4_CEandDiceloss_fold0.pth'
# model1_path = './checkpoints/task1/train_size800_batchsize4_CEandDiceloss_fold1.pth'
# model2_path = './checkpoints/task1/train_size800_batchsize4_CEandDiceloss_fold2.pth'
# model3_path = './checkpoints/task1/train_size800_batchsize4_CEandDiceloss_fold3.pth'
# model4_path = './checkpoints/task1/train_size800_batchsize4_CEandDiceloss_fold4.pth'
# image_size = 800
# test_root = '../datasets/Validation/Image'

# model0 = UNet()
# para_state_dict = torch.load(model0_path)
# model0.load_state_dict(para_state_dict)
# model0.eval()

# model1 = UNet()
# para_state_dict = torch.load(model1_path)
# model1.load_state_dict(para_state_dict)
# model1.eval()

# model2 = UNet()
# para_state_dict = torch.load(model2_path)
# model2.load_state_dict(para_state_dict)
# model2.eval()

# model3 = UNet()
# para_state_dict = torch.load(model3_path)
# model3.load_state_dict(para_state_dict)
# model3.eval()

# model4 = UNet()
# para_state_dict = torch.load(model4_path)
# model4.load_state_dict(para_state_dict)
# model4.eval()

# img_test_transforms = trans.Compose([
#     trans.ToTensor(),
#     trans.Resize((image_size, image_size))
# ])

# test_dataset = OCTDataset(image_transforms=img_test_transforms,
#                         image_path=test_root,
#                         mode='test')
# cache = []
# for img, idx, h, w in test_dataset:
#     # print('./submission/task1/Layer_Segmentations-val/'+idx)
#     img = img.unsqueeze(0) # 增加维度：(3,256,256)-->(1,3,256,256)
#     logits0 = model0(img) # 模型输出维度(1,4,256,256)
#     m = torch.nn.Softmax(dim=1)
#     logits0 = m(logits0)
    
#     logits1 = model1(img) # 模型输出维度(1,4,256,256)
#     logits1 = m(logits1)

#     logits2 = model2(img)
#     logits2 = m(logits2)

#     logits3 = model3(img)
#     logits3 = m(logits3)

#     logits4 = model4(img)
#     logits4 = m(logits4)

#     logits = (logits0+logits1+logits2+logits3+logits4)/5

#     pred_img = logits.detach().numpy().argmax(1)
#     pred_gray = np.squeeze(pred_img, axis=0)
#     pred_gray = pred_gray.astype('float32')

#     pred_gray[pred_gray == 1] = 80
#     pred_gray[pred_gray == 2] = 160
#     pred_gray[pred_gray == 3] = 255
#     pred = cv2.resize(pred_gray, (w, h))
#     print(idx)
#     cv2.imwrite('./submission/task1/Layer_Segmentations-val/'+idx, pred)




# # 预测阶段
# model0_path = './checkpoints/task1/train_size800_batchsize4_aug_CEandDiceloss_fold0.pth'
# image_size = 800
# test_root = '../datasets/Validation/Image'
# # os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model0 = UNet()
# # model0 = nn.DataParallel(model0).to(device) 
# para_state_dict = torch.load(model0_path)
# model0.load_state_dict(para_state_dict)

# model0.eval()


# img_test_transforms = trans.Compose([
#     trans.ToTensor(),
#     trans.Resize((image_size, image_size))
# ])

# test_dataset = OCTDataset(image_transforms=img_test_transforms,
#                         image_path=test_root,
#                         mode='test')
# cache = []
# for img, idx, h, w in test_dataset:
#     # print('./submission/task1/Layer_Segmentations-val/'+idx)
#     img = img.unsqueeze(0) # 增加维度：(3,256,256)-->(1,3,256,256)
#     logits0 = model0(img) # 模型输出维度(1,4,256,256)
#     m = torch.nn.Softmax(dim=1)
#     logits = m(logits0)

#     pred_img = logits.detach().cpu().numpy().argmax(1) # pred_img维度1*512*512
#     pred_gray = np.squeeze(pred_img, axis=0)
#     pred_gray = pred_gray.astype('float32')

#     pred_gray[pred_gray == 1] = 80
#     pred_gray[pred_gray == 2] = 160
#     pred_gray[pred_gray == 3] = 255
#     pred = cv2.resize(pred_gray, (w, h))  # pred大小800*1100
#     print(h, w)  # h=800, w=1100
#     cv2.imwrite('./submission/task1/Layer_Segmentations-val/'+idx, pred)



# # 使用augmentation模型预测
# model0_path = './checkpoints/task1/train_transunet_size800_batchsize4_aug4_CEandDiceloss_fold0.pth'
# image_size = 800
# test_root1 = '../datasets/Validation/Image_aug2_1'
# test_root2 = '../datasets/Validation/Image_aug2_2'

# # model0 = UNet()
# # model0 = CE_Net_()
# model0 = R50_ViT_B_16(in_ch=3,out_ch=4)
# para_state_dict = torch.load(model0_path)
# model0.load_state_dict(para_state_dict)
# model0.eval()

# # img_test_transforms = trans.Compose([
# #     trans.ToTensor(),
# # ])
# img_test_transforms = img_transforms(applied_types='test')

# test_dataset1 = OCTDataset(image_transforms=img_test_transforms,
#                         image_path=test_root1,
#                         mode='test')

# test_dataset2 = OCTDataset(image_transforms=img_test_transforms,
#                         image_path=test_root2,
#                         mode='test')

# cache = []
# for img1, idx1, h, w in test_dataset1:

#     for img2, idx2, h, w in test_dataset2:
#         if idx2 == idx1:
#             img1 = img1.unsqueeze(0) # 增加维度：(3,800,800)-->(1,3,800,800)
#             logits1 = model0(img1) # 模型输出维度(1,4,800,800)
#             img2 = img2.unsqueeze(0)
#             logits2 = model0(img2)

#             m = torch.nn.Softmax(dim=1)
#             logits1 = m(logits1).detach().numpy()
#             logits2 = m(logits2).detach().numpy()

#             overlap1 = logits1[:,:,:,300:800]
#             overlap2 = logits2[:,:,:,0:500]
#             overlap = (overlap1+overlap2)/2

#             logits = np.concatenate((logits1[:,:,:,0:300], overlap, logits2[:,:,:,500:800]), axis=3)


#             pred_img = logits.argmax(1)
#             pred_gray = np.squeeze(pred_img, axis=0)
#             pred_gray = pred_gray.astype('float32')

#             pred_gray[pred_gray == 1] = 80
#             pred_gray[pred_gray == 2] = 160
#             pred_gray[pred_gray == 3] = 255
#             # pred = cv2.resize(pred_gray, (w, h))
#             # print(pred_gray.shape)
#             print(idx1)
#             cv2.imwrite('./submission/task1/Layer_Segmentations-val/'+idx1, pred_gray)



# 使用augmentation模型预测, 五折交叉验证
model0_path = './checkpoints/task1/cenetECA_bestpseudolabel_pickout_imgaugmore_trainvalClearlySplit_focalCEandDiceloss_fold0.pth'
model1_path = './checkpoints/task1/cenetECA_bestpseudolabel_pickout_imgaugmore_trainvalClearlySplit_focalCEandDiceloss_fold1.pth'
model2_path = './checkpoints/task1/cenetECA_bestpseudolabel_pickout_imgaugmore_trainvalClearlySplit_focalCEandDiceloss_fold2.pth'
model3_path = './checkpoints/task1/cenetECA_bestpseudolabel_pickout_imgaugmore_trainvalClearlySplit_focalCEandDiceloss_fold3.pth'
model4_path = './checkpoints/task1/cenetECA_bestpseudolabel_pickout_imgaugmore_trainvalClearlySplit_focalCEandDiceloss_fold4.pth'
image_size = 800
test_root1 = '../datasets/Validation/Image_aug2_1'
test_root2 = '../datasets/Validation/Image_aug2_2'
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# model0 = UNet()
# model0 = CE_Net_()
# model0 = CE_Net_withSE()
model0 = CE_Net_withECA().to(device)
# model0 = CE_Net_SEwoReduction()
# model0 = R50_ViT_B_16(in_ch=3,out_ch=4)
# model0 = MSNet().to(device)
# para_state_dict = torch.load(model0_path,map_location={'cuda:1': 'cuda:1'})
para_state_dict = torch.load(model0_path)
model0.load_state_dict(para_state_dict)
model0.eval()

# model1 = UNet()
# model1 = CE_Net_()
# model1 = CE_Net_withSE()
model1 = CE_Net_withECA().to(device)
# model1 = CE_Net_SEwoReduction()
# model1 = R50_ViT_B_16(in_ch=3,out_ch=4)
# model1 = MSNet().to(device)
para_state_dict = torch.load(model1_path)
model1.load_state_dict(para_state_dict)
model1.eval()

# model2 = UNet()
# model2 = CE_Net_()
# model2 = CE_Net_withSE()
model2 = CE_Net_withECA().to(device)
# model2 = CE_Net_SEwoReduction()
# model2 = R50_ViT_B_16(in_ch=3,out_ch=4)
# model2 = MSNet().to(device)
para_state_dict = torch.load(model2_path)
model2.load_state_dict(para_state_dict)
model2.eval()

# model3 = UNet()
# model3 = CE_Net_()
# model3 = CE_Net_withSE()
model3 = CE_Net_withECA().to(device)
# model3 = CE_Net_SEwoReduction()
# model3 = R50_ViT_B_16(in_ch=3,out_ch=4)
# model3 = MSNet().to(device)
para_state_dict = torch.load(model3_path)
model3.load_state_dict(para_state_dict)
model3.eval()

# model4 = UNet()
# model4 = CE_Net_()
# model4 = CE_Net_withSE()
model4 = CE_Net_withECA().to(device)
# model4 = CE_Net_SEwoReduction()
# model4 = R50_ViT_B_16(in_ch=3,out_ch=4)
# model4 = MSNet().to(device)
para_state_dict = torch.load(model4_path)
model4.load_state_dict(para_state_dict)
model4.eval()

# img_test_transforms = trans.Compose([
#     trans.ToTensor(),
# ])
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
            img1 = img1.to(device)
            img2 = img2.to(device)
            img1 = img1.unsqueeze(0) # 增加维度：(3,800,800)-->(1,3,800,800)
            logits1 = (model0(img1)+model1(img1)+model2(img1)+model3(img1)+model4(img1))/5 # 模型输出维度(1,4,800,800)
            img2 = img2.unsqueeze(0)
            logits2 = (model0(img2)+model1(img2)+model2(img2)+model3(img2)+model4(img2))/5

            m = torch.nn.Softmax(dim=1)
            logits1 = m(logits1).cpu().detach().numpy()
            logits2 = m(logits2).cpu().detach().numpy()

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
            cv2.imwrite('./submission/task1/cenetECA_bestpseudolabel_pickout_imgaugmore_trainvalClearlySplit/'+idx1, pred_gray)



# # 测试图像拼接, 两张550*550拼成一张800*1100
# image_root_1 = './submission/task1/Layer_Segmentations_crop/1'
# image_root_2 = './submission/task1/Layer_Segmentations_crop/2'
# filelist_1 = os.listdir(image_root_1)
# filelist_2 = os.listdir(image_root_2)

# for img_idx in filelist_1:
#     print(img_idx)
#     crop1 = cv2.imread(os.path.join(image_root_1, img_idx))
#     crop2 = cv2.imread(os.path.join(image_root_2, img_idx))
#     image = np.concatenate((crop1, crop2), axis = 1)
#     image = cv2.copyMakeBorder(image, 125, 125, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
#     # print(image)
#     cv2.imwrite('./submission/task1/Layer_Segmentations-val/'+img_idx, image)
    


