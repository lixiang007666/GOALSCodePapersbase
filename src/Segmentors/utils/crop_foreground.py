import numpy as np
import os
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from config import duts,ecssd,hku_is,dut_omron,pascal_s,CAMO,CHAMELEON,COD10K,NC4K,CVC_300,CVC_ClinicDB,CVC_ColonDB,ETIS_LaribPolypDB,Kvasir,SBU,ucf,CUHK,DUT,ISTD,dutrgbd,njud,nlpr,stere,sip,rgbd135,ssd,lfsd,trans10k_easy,trans10k_hard,trans10k_all,ORSSD,EORSSD,ORSI_4199,GDD,MSD,weixin_data
from misc import check_mkdir,crf_refine
import ttach as tta
import cv2
torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = '/home/asus/'
# exp_name = 'Coding/weihong_preoject/saved_model'
exp_name = 'Coding/qingguangyan_miccai_2022_workshop/saved_model'
args = {
    'snapshot1': 'GOALS_challenges_miccai2022_f3net_strategy_mask_1_GateNet_dem_512_with_foldaspp_gap_resnext101_32x8d_datasets_rgb_strategy2_w_h_1100_800_size_noms_nocolorEnhance_batch2_train_50epochModel_49_gen',
    'snapshot2': 'GOALS_challenges_miccai2022_mask_1_2_GateNet_dem_512_with_foldaspp_gap_resnext101_32x8d_datasets_rgb_strategy2_w_h_1100_800_size_noms_nocolorEnhance_batch2_train_100epochModel_100_gen',
    'snapshot3': 'GOALS_challenges_miccai2022_mask_3_GateNet_dem_128_with_foldaspp_gap_resnext101_32x8d_datasets_rgb_strategy2_w_h_1100_800_size_noms_batch2_train_100epochModel_100_gen',
    # 'snapshot3': 'GOALS_challenges_miccai2022_mask_3_gatenetv2_resnext101_datasets_rgb_strategy2_w_h_1100_800_size_noms_batch2train_100epochModel_100_gen',
    'crf_refine': True,
    'save_results': True
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

depth_transform = transforms.ToTensor()
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

to_test = {'crop_train':weixin_data}
transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        # tta.Scale(scales=[0.75, 1, 1.25], interpolation='bilinear', align_corners=False),
        tta.Scale(scales=[1], interpolation='bilinear', align_corners=False),
    ]
)
task = 'challenge'

def main():
    with torch.no_grad():
        for name, root in to_test.items():
            check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot1'])))
            root1 = os.path.join(root,'Image')
            img_list = [os.path.splitext(f) for f in os.listdir(root1) if f.endswith('.png')]
            print(len(img_list))
            for idx, img_name in enumerate(img_list):
                print(img_name[0])
                # print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                rgb_png_path = os.path.join(root, 'Image', img_name[0] + '.png')
                rgb_png_path_mask_1_2_3 = os.path.join(root, 'Layer_Masks', img_name[0] + '.png')
                rgb_png_path_mask_1 = os.path.join(root, 'mask_1', img_name[0] + '.png')
                rgb_png_path_mask_1_2 = os.path.join(root, 'mask_1_2', img_name[0] + '.png')
                rgb_png_path_mask_3 = os.path.join(root, 'mask_3', img_name[0] + '.png')
                img = Image.open(rgb_png_path).convert('RGB')
                img_np = np.array(img)
                mask_1_2_3 = Image.open(rgb_png_path_mask_1_2_3).convert('L')
                mask_1_2_3_np = np.array(mask_1_2_3)/255
                mask_1 = Image.open(rgb_png_path_mask_1).convert('L')
                mask_1_np = np.array(mask_1)/255
                mask_1_2 = Image.open(rgb_png_path_mask_1_2).convert('L')
                mask_1_2_np = np.array(mask_1_2)/255
                mask_3 = Image.open(rgb_png_path_mask_3).convert('L')
                mask_3_np = np.array(mask_3)/255
                # print(mask_1_2_3_np.max())
                #
                # print(mask_1_2_3_np.shape)
                crop_index = []
                for i in range(800):
                    if sum(mask_1_2_3_np[i,0:1100])!=1100:
                        crop_index.append(i)
                start_crop = crop_index[0]
                end_crop = crop_index[-1]
                print(start_crop,end_crop)
                crop_mask = np.zeros((mask_1_2_3_np.shape[0], mask_1_2_3_np.shape[1]), np.uint8)
                crop_mask[start_crop-10:end_crop+10,0:1100] = 255
                #


                check_mkdir(os.path.join(ckpt_path, exp_name,args['snapshot1']+'epoch',task,name))
                # cv2.imwrite(os.path.join(ckpt_path, exp_name ,args['snapshot1']+'epoch',task,name, img_name[0] + '_img.png'), img_np[start_crop-10:end_crop+10,0:1100])
                # cv2.imwrite(os.path.join(ckpt_path, exp_name ,args['snapshot1']+'epoch',task,name, img_name[0] + '_mask1.png'), mask_1_np[start_crop-10:end_crop+10,0:1100]*255)
                # cv2.imwrite(os.path.join(ckpt_path, exp_name ,args['snapshot1']+'epoch',task,name, img_name[0] + '_mask1_2.png'), mask_1_2_np[start_crop-10:end_crop+10,0:1100]*255)
                # cv2.imwrite(os.path.join(ckpt_path, exp_name ,args['snapshot1']+'epoch',task,name, img_name[0] + '_mask3.png'), mask_3_np[start_crop-10:end_crop+10,0:1100]*255)
                cv2.imwrite(os.path.join(ckpt_path, exp_name ,args['snapshot1']+'epoch',task,name, img_name[0] + '_crop_mask.png'), crop_mask)
                # # prediction.save(os.path.join(ckpt_path, exp_name ,args['snapshot']+'epoch',name, img_name[0] + '.png'))
                # prediction.save(os.path.join(ckpt_path, exp_name ,args['snapshot']+'epoch',name, img_name[0] + '.jpg'))




if __name__ == '__main__':
    main()
