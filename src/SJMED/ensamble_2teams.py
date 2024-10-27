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

device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device2 = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
device3 = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
device4 = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
device5 = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
device6 = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
device7 = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')




model0_path = './checkpoints/task1/cenetECA_labelsmooth_CEandDiceloss_fold0.pth'
model1_path = './checkpoints/task1/cenetECA_labelsmooth_CEandDiceloss_fold1.pth'
model2_path = './checkpoints/task1/cenetECA_labelsmooth_CEandDiceloss_fold2.pth'
model3_path = './checkpoints/task1/cenetECA_labelsmooth_CEandDiceloss_fold3.pth'
model4_path = './checkpoints/task1/cenetECA_labelsmooth_CEandDiceloss_fold4.pth'
model5_path = './checkpoints/task1/cenetECA_pseudolabel_imgaugmore_trainvalClearlySplit_focalCEandDiceloss_fold0.pth'
model6_path = './checkpoints/task1/cenetECA_pseudolabel_imgaugmore_trainvalClearlySplit_focalCEandDiceloss_fold1.pth'
model7_path = './checkpoints/task1/cenetECA_pseudolabel_imgaugmore_trainvalClearlySplit_focalCEandDiceloss_fold2.pth'
model8_path = './checkpoints/task1/cenetECA_pseudolabel_imgaugmore_trainvalClearlySplit_focalCEandDiceloss_fold3.pth'
model9_path = './checkpoints/task1/cenetECA_pseudolabel_imgaugmore_trainvalClearlySplit_focalCEandDiceloss_fold4.pth'
model10_path = './checkpoints/task1/cenetECA_bestpseudolabel_pickout_imgaugmore_trainvalClearlySplit_focalCEandDiceloss_fold0.pth'
model11_path = './checkpoints/task1/cenetECA_bestpseudolabel_pickout_imgaugmore_trainvalClearlySplit_focalCEandDiceloss_fold1.pth'
model12_path = './checkpoints/task1/cenetECA_bestpseudolabel_pickout_imgaugmore_trainvalClearlySplit_focalCEandDiceloss_fold2.pth'
model13_path = './checkpoints/task1/cenetECA_bestpseudolabel_pickout_imgaugmore_trainvalClearlySplit_focalCEandDiceloss_fold3.pth'
model14_path = './checkpoints/task1/cenetECA_bestpseudolabel_pickout_imgaugmore_trainvalClearlySplit_focalCEandDiceloss_fold4.pth'
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
model30_path = './checkpoints/task1/DeepLabV3Plus_labelsmooth0.05/train_DeepLabV3Plus_CEDiceFocallovasz_fold0.pth'
model31_path = './checkpoints/task1/DeepLabV3Plus_labelsmooth0.05/train_DeepLabV3Plus_CEDiceFocallovasz_fold1.pth'
model32_path = './checkpoints/task1/DeepLabV3Plus_labelsmooth0.05/train_DeepLabV3Plus_CEDiceFocallovasz_fold2.pth'
model33_path = './checkpoints/task1/DeepLabV3Plus_labelsmooth0.05/train_DeepLabV3Plus_CEDiceFocallovasz_fold3.pth'
model34_path = './checkpoints/task1/DeepLabV3Plus_labelsmooth0.05/train_DeepLabV3Plus_CEDiceFocallovasz_fold4.pth'
model35_path = './checkpoints/task1/DeepLabV3Plus_labelsmooth0.1/train_DeepLabV3Plus_CEDiceFocallovasz_fold0.pth'
model36_path = './checkpoints/task1/DeepLabV3Plus_labelsmooth0.1/train_DeepLabV3Plus_CEDiceFocallovasz_fold1.pth'
model37_path = './checkpoints/task1/DeepLabV3Plus_labelsmooth0.1/train_DeepLabV3Plus_CEDiceFocallovasz_fold2.pth'
model38_path = './checkpoints/task1/DeepLabV3Plus_labelsmooth0.1/train_DeepLabV3Plus_CEDiceFocallovasz_fold3.pth'
model39_path = './checkpoints/task1/DeepLabV3Plus_labelsmooth0.1/train_DeepLabV3Plus_CEDiceFocallovasz_fold4.pth'
model40_path = './checkpoints/task1/Unet/train_UNet_CEandDiceloss_fold0.pth'
model41_path = './checkpoints/task1/Unet/train_UNet_CEandDiceloss_fold1.pth'
model42_path = './checkpoints/task1/Unet/train_UNet_CEandDiceloss_fold2.pth'
model43_path = './checkpoints/task1/Unet/train_UNet_CEandDiceloss_fold3.pth'
model44_path = './checkpoints/task1/Unet/train_UNet_CEandDiceloss_fold4.pth'



image_size = 800
test_root1 = '../datasets/Validation/Image_aug2_1'
test_root2 = '../datasets/Validation/Image_aug2_2'


model0 = CE_Net_withECA().to(device0)
para_state_dict = torch.load(model0_path)
model0.load_state_dict(para_state_dict)
model0.eval()

model1 = CE_Net_withECA().to(device0)
para_state_dict = torch.load(model1_path)
model1.load_state_dict(para_state_dict)
model1.eval()

model2 = CE_Net_withECA().to(device0)
para_state_dict = torch.load(model2_path)
model2.load_state_dict(para_state_dict)
model2.eval()

model3 = CE_Net_withECA().to(device0)
para_state_dict = torch.load(model3_path)
model3.load_state_dict(para_state_dict)
model3.eval()

model4 = CE_Net_withECA().to(device0)
para_state_dict = torch.load(model4_path)
model4.load_state_dict(para_state_dict)
model4.eval()

model5 = CE_Net_withECA().to(device0)
para_state_dict = torch.load(model5_path)
model5.load_state_dict(para_state_dict)
model5.eval()

model6 = CE_Net_withECA().to(device0)
para_state_dict = torch.load(model6_path)
model6.load_state_dict(para_state_dict)
model6.eval()

model7 = CE_Net_withECA().to(device0)
para_state_dict = torch.load(model7_path)
model7.load_state_dict(para_state_dict)
model7.eval()

model8 = CE_Net_withECA().to(device0)
para_state_dict = torch.load(model8_path)
model8.load_state_dict(para_state_dict)
model8.eval()

model9 = CE_Net_withECA().to(device0)
para_state_dict = torch.load(model9_path)
model9.load_state_dict(para_state_dict)
model9.eval()

model10 = CE_Net_withECA().to(device1)
para_state_dict = torch.load(model10_path)
model10.load_state_dict(para_state_dict)
model10.eval()

model11 = CE_Net_withECA().to(device1)
para_state_dict = torch.load(model11_path)
model11.load_state_dict(para_state_dict)
model11.eval()

model12 = CE_Net_withECA().to(device1)
para_state_dict = torch.load(model12_path)
model12.load_state_dict(para_state_dict)
model12.eval()

model13 = CE_Net_withECA().to(device1)
para_state_dict = torch.load(model13_path)
model13.load_state_dict(para_state_dict)
model13.eval()

model14 = CE_Net_withECA().to(device1)
para_state_dict = torch.load(model14_path)
model14.load_state_dict(para_state_dict)
model14.eval()

model15 = MSNet().to(device3)
para_state_dict = torch.load(model15_path)
model15.load_state_dict(para_state_dict)
model15.eval()

model16 = MSNet().to(device3)
para_state_dict = torch.load(model16_path)
model16.load_state_dict(para_state_dict)
model16.eval()

model17 = MSNet().to(device3)
para_state_dict = torch.load(model17_path)
model17.load_state_dict(para_state_dict)
model17.eval()

model18 = MSNet().to(device3)
para_state_dict = torch.load(model18_path)
model18.load_state_dict(para_state_dict)
model18.eval()

model19 = MSNet().to(device3)
para_state_dict = torch.load(model19_path)
model19.load_state_dict(para_state_dict)
model19.eval()

model20 = CE_Net_withECA_masklayer().to(device2)
para_state_dict = torch.load(model20_path)
model20.load_state_dict(para_state_dict)
model20.eval()

model21 = CE_Net_withECA_masklayer().to(device2)
para_state_dict = torch.load(model21_path)
model21.load_state_dict(para_state_dict)
model21.eval()

model22 = CE_Net_withECA_masklayer().to(device2)
para_state_dict = torch.load(model22_path)
model22.load_state_dict(para_state_dict)
model22.eval()

model23 = CE_Net_withECA_masklayer().to(device2)
para_state_dict = torch.load(model23_path)
model23.load_state_dict(para_state_dict)
model23.eval()

model24 = CE_Net_withECA_masklayer().to(device2)
para_state_dict = torch.load(model24_path)
model24.load_state_dict(para_state_dict)
model24.eval()

model25 = CE_Net_withECA_masklayer().to(device2)
para_state_dict = torch.load(model25_path)
model25.load_state_dict(para_state_dict)
model25.eval()

model26 = CE_Net_withECA_masklayer().to(device2)
para_state_dict = torch.load(model26_path)
model26.load_state_dict(para_state_dict)
model26.eval()

model27 = CE_Net_withECA_masklayer().to(device2)
para_state_dict = torch.load(model27_path)
model27.load_state_dict(para_state_dict)
model27.eval()

model28 = CE_Net_withECA_masklayer().to(device2)
para_state_dict = torch.load(model28_path)
model28.load_state_dict(para_state_dict)
model28.eval()

model29 = CE_Net_withECA_masklayer().to(device2)
para_state_dict = torch.load(model29_path)
model29.load_state_dict(para_state_dict)
model29.eval()

# model30 = smp.DeepLabV3Plus(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# ).to(device3)
# para_state_dict = torch.load(model30_path)
# model30.load_state_dict(para_state_dict)
# model30.eval()

# model31 = smp.DeepLabV3Plus(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# ).to(device3)
# para_state_dict = torch.load(model31_path)
# model31.load_state_dict(para_state_dict)
# model31.eval()

# model32 = smp.DeepLabV3Plus(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# ).to(device4)
# para_state_dict = torch.load(model32_path)
# model32.load_state_dict(para_state_dict)
# model32.eval()

# model33 = smp.DeepLabV3Plus(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# ).to(device4)
# para_state_dict = torch.load(model33_path)
# model33.load_state_dict(para_state_dict)
# model33.eval()

# model34 = smp.DeepLabV3Plus(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# ).to(device5)
# para_state_dict = torch.load(model34_path)
# model34.load_state_dict(para_state_dict)
# model34.eval()


# model35 = smp.DeepLabV3Plus(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# ).to(device5)
# para_state_dict = torch.load(model35_path)
# model35.load_state_dict(para_state_dict)
# model35.eval()

# model36 = smp.DeepLabV3Plus(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# ).to(device6)
# para_state_dict = torch.load(model36_path)
# model36.load_state_dict(para_state_dict)
# model36.eval()

# model37 = smp.DeepLabV3Plus(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# ).to(device6)
# para_state_dict = torch.load(model37_path)
# model37.load_state_dict(para_state_dict)
# model37.eval()

# model38 = smp.DeepLabV3Plus(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# ).to(device7)
# para_state_dict = torch.load(model38_path)
# model38.load_state_dict(para_state_dict)
# model38.eval()

# model39 = smp.DeepLabV3Plus(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# ).to(device7)
# para_state_dict = torch.load(model39_path)
# model39.load_state_dict(para_state_dict)
# model39.eval()

# model40 = smp.Unet(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# )
# para_state_dict = torch.load(model40_path)
# model40.load_state_dict(para_state_dict)
# model40.eval()

# model41 = smp.Unet(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# )
# para_state_dict = torch.load(model41_path)
# model41.load_state_dict(para_state_dict)
# model41.eval()

# model42 = smp.Unet(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# )
# para_state_dict = torch.load(model42_path)
# model42.load_state_dict(para_state_dict)
# model42.eval()

# model43 = smp.Unet(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# )
# para_state_dict = torch.load(model43_path)
# model43.load_state_dict(para_state_dict)
# model43.eval()

# model44 = smp.Unet(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# )
# para_state_dict = torch.load(model44_path)
# model44.load_state_dict(para_state_dict)
# model44.eval()



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
            for i in range(10):
                img1 = img1.to(device0)
                img2 = img2.to(device0)
                logits1 += locals()['model'+str(i)](img1).to(device0)
                logits2 += locals()['model'+str(i)](img2).to(device0)
            for i in range(10,15):
                img1 = img1.to(device1)
                img2 = img2.to(device1)
                logits1 = logits1.to(device1)
                logits2 = logits2.to(device1)
                logits1 += locals()['model'+str(i)](img1)
                logits2 += locals()['model'+str(i)](img2)
            for i in range(15,20):
                img1 = img1.to(device3)
                img2 = img2.to(device3)
                logits1 = logits1.to(device3)
                logits2 = logits2.to(device3)
                logits1 += locals()['model'+str(i)](img1)
                logits2 += locals()['model'+str(i)](img2)
            
            for i in range(20,30):
                img1 = img1.to(device2)
                img2 = img2.to(device2)
                logits1 = logits1.to(device2)
                logits2 = logits2.to(device2)
                logits1 += locals()['model'+str(i)](img1)[:,0:4,:,:]
                logits2 += locals()['model'+str(i)](img2)[:,0:4,:,:]
            # for i in range(30,32):
            #     img1 = img1.to(device3)
            #     img2 = img2.to(device3)
            #     logits1 = logits1.to(device3)
            #     logits2 = logits2.to(device3)
            #     logits1 += locals()['model'+str(i)](img1)
            #     logits2 += locals()['model'+str(i)](img2)
            # for i in range(32,34):
            #     img1 = img1.to(device4)
            #     img2 = img2.to(device4)
            #     logits1 = logits1.to(device4)
            #     logits2 = logits2.to(device4)
            #     logits1 += locals()['model'+str(i)](img1)
            #     logits2 += locals()['model'+str(i)](img2)
            # for i in range(34,36):
            #     img1 = img1.to(device5)
            #     img2 = img2.to(device5)
            #     logits1 = logits1.to(device5)
            #     logits2 = logits2.to(device5)
            #     logits1 += locals()['model'+str(i)](img1)
            #     logits2 += locals()['model'+str(i)](img2)
            # for i in range(36,38):
            #     img1 = img1.to(device6)
            #     img2 = img2.to(device6)
            #     logits1 = logits1.to(device6)
            #     logits2 = logits2.to(device6)
            #     logits1 += locals()['model'+str(i)](img1)
            #     logits2 += locals()['model'+str(i)](img2)
            # for i in range(38,40):
            #     img1 = img1.to(device7)
            #     img2 = img2.to(device7)
            #     logits1 = logits1.to(device7)
            #     logits2 = logits2.to(device7)
            #     logits1 += locals()['model'+str(i)](img1)
            #     logits2 += locals()['model'+str(i)](img2)
            
            # img1 = img1.to(device5)
            # img2 = img2.to(device5)
            # logits1 = logits1.to(device5)
            # logits2 = logits2.to(device5)
            # logits1 += locals()['model'+str(17)](img1)
            # logits2 += locals()['model'+str(17)](img2)

            # img1 = img1.to(device6)
            # img2 = img2.to(device6)
            # logits1 = logits1.to(device6)
            # logits2 = logits2.to(device6)
            # logits1 += locals()['model'+str(18)](img1)
            # logits2 += locals()['model'+str(18)](img2)

            # img1 = img1.to(device7)
            # img2 = img2.to(device7)
            # logits1 = logits1.to(device7)
            # logits2 = logits2.to(device7)
            # logits1 += locals()['model'+str(19)](img1)
            # logits2 += locals()['model'+str(19)](img2)
            
            logits1 = (logits1/40).cpu()
            logits2 = (logits2/40).cpu()
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
            cv2.imwrite('./submission/task1/ensamble7_trainvalClearlySplit_mine/'+idx1, pred_gray)




# model35 = smp.DeepLabV3Plus(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# )
# para_state_dict = torch.load(model35_path)
# model35.load_state_dict(para_state_dict)
# model35.eval()

# model36 = smp.DeepLabV3Plus(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# )
# para_state_dict = torch.load(model36_path)
# model36.load_state_dict(para_state_dict)
# model36.eval()

# model37 = smp.DeepLabV3Plus(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# )
# para_state_dict = torch.load(model37_path)
# model37.load_state_dict(para_state_dict)
# model37.eval()

# model38 = smp.DeepLabV3Plus(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# )
# para_state_dict = torch.load(model38_path)
# model38.load_state_dict(para_state_dict)
# model38.eval()

# model39 = smp.DeepLabV3Plus(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# )
# para_state_dict = torch.load(model39_path)
# model39.load_state_dict(para_state_dict)
# model39.eval()


# model40 = smp.Unet(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# )
# para_state_dict = torch.load(model40_path)
# model40.load_state_dict(para_state_dict)
# model40.eval()

# model41 = smp.Unet(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# )
# para_state_dict = torch.load(model41_path)
# model41.load_state_dict(para_state_dict)
# model41.eval()

# model42 = smp.Unet(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# )
# para_state_dict = torch.load(model42_path)
# model42.load_state_dict(para_state_dict)
# model42.eval()

# model43 = smp.Unet(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# )
# para_state_dict = torch.load(model43_path)
# model43.load_state_dict(para_state_dict)
# model43.eval()

# model44 = smp.Unet(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=4,                      # model output channels (number of classes in your dataset)
# )
# para_state_dict = torch.load(model44_path)
# model44.load_state_dict(para_state_dict)
# model44.eval()




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
