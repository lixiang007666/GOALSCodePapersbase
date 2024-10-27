import numpy as np
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from utils.config import test_data
from utils.misc import check_mkdir,crf_refine
from model.gatenet_rgb_res50_pami import GateNet_dem_512
from model.miccai_msnet import MSNet
from model.minet import MINet_Resnet
import ttach as tta
import torch.nn.functional as F
import cv2
torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './'
exp_name = 'saved_model'
args = {
    'snapshot1': 'GOALS_challenges_miccai2022_mask_1_GateNet_dem_512_with_foldaspp_gap_resnext101_32x8d_datasets_rgb_strategy2_w_h_1100_800_size_noms_nocolorEnhance_batch2_train_300epochModel_300_gen',
    'snapshot2': 'GOALS_challenges_miccai2022_mask_1_2_GateNet_dem_512_with_foldaspp_gap_resnext101_32x8d_datasets_rgb_strategy2_w_h_1100_800_size_noms_nocolorEnhance_batch2_train_300epochModel_300_gen',
    'snapshot3': 'GOALS_challenges_miccai2022_mask_3_GateNet_dem_512_with_foldaspp_gap_resnext101_32x8d_datasets_rgb_strategy2_w_h_880_640_size_ms_batch2_train_100epochModel_100_gen',
    'snapshot6': 'GOALS_challenges_miccai2022_mask_3_MINet_Resnet_with_foldaspp_gap_resnext101_32x8d_datasets_rgb_strategy2_w_h_880_640_size_ms_batch4_train_100epoch_add_all_dataenhancedModel_100_gen',
    'snapshot7': 'GOALS_challenges_miccai2022_mask_1_MSNet_foldaspp_resnext101_32x8d_datasets_rgb_strategy2_w_h_1100_800_size_noms_all_enhanced_batch4_train_300epochModel_300_gen',
    'snapshot8': 'GOALS_challenges_miccai2022_mask_1_2_MSNet_foldaspp_resnext101_32x8d_datasets_rgb_strategy2_w_h_1100_800_size_noms_all_enhanced_batch4_train_300epochModel_300_gen',
    'snapshot15': 'GOALS_challenges_miccai2022_mask_3_MSNet_with_foldaspp_gap_resnext101_32x8d_datasets_rgb_strategy2_w_h_880_640_size_ms_batch4_train_200epoch_add_all_dataenhancedModel_200_gen',
    'snapshot18': 'GOALS_challenges_miccai2022_mask_3_MSNet_with_foldaspp_gap_resnext101_32x8d_datasets_rgb_strategy2_w_h_880_640_size_ms_batch4_train_100epoch_add_all_dataenhancedModel_100_gen',
    'snapshot23': 'GOALS_challenges_miccai2022_mask_1_MSNet_with_foldaspp_gap_resnext101_32x8d_datasets_rgb_strategy2_w_h_880_640_size_ms_batch4_train_300epoch_add_all_dataenhanceModel_300_gen',
    'snapshot24': 'GOALS_challenges_miccai2022_mask_1_2_MSNet_with_foldaspp_gap_resnext101_32x8d_datasets_rgb_strategy2_w_h_880_640_size_ms_batch4_train_300epoch_add_all_dataenhanceModel_300_gen',
    'snapshot25': 'GOALS_challenges_miccai2022_mask_1_MINet_Resnet_with_foldaspp_gap_resnext101_32x8d_datasets_rgb_strategy2_w_h_880_640_size_ms_batch4_train_300epoch_add_all_dataenhancedModel_300_gen',
    'snapshot26': 'GOALS_challenges_miccai2022_mask_1_2_MINet_Resnet_with_foldaspp_gap_resnext101_32x8d_datasets_rgb_strategy2_w_h_880_640_size_ms_batch4_train_300epoch_add_all_dataenhancedModel_300_gen',
    'snapshot29': 'GOALS_challenges_miccai2022_mask_1_GateNet_dem_512_with_foldaspp_gap_resnext101_32x8d_datasets_rgb_strategy2_w_h_880_640_size_ms_add_all_dataenhanced_batch2_train_300epochModel_300_gen',
    'snapshot30': 'GOALS_challenges_miccai2022_mask_1_2_GateNet_dem_512_with_foldaspp_gap_resnext101_32x8d_datasets_rgb_strategy2_w_h_880_640_size_ms_add_all_dataenhanced_batch2_train_300epochModel_300_gen',
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

to_test = {'Layer_Segmentations':test_data}
transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Scale(scales=[1], interpolation='bilinear', align_corners=False),
    ]
)
transforms_new = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Scale(scales=[0.75, 1, 1.25], interpolation='bilinear', align_corners=False),
    ]
)

task = 'challenge'

def filter_ring(rectangle_img_np,filter_thr):
    rectangle_img_np = np.array(rectangle_img_np, dtype="uint8")
    rectangle_img_np = rectangle_img_np//255
    num_labels, labels, stats_rectangle, centroids = cv2.connectedComponentsWithStats(rectangle_img_np, ltype=cv2.CV_32S)
    area = []
    real_index = []
    for i in stats_rectangle:
        area.append(i[-1])
    if len(area)==1:
        flag_ring = 0
    else:
        flag_ring = 1
        max_area = max(area[1:])
        for i, v in enumerate(area):
            if v >= max_area/filter_thr:
                real_index.append(i)
    rectangle_single = np.zeros((rectangle_img_np.shape[0], rectangle_img_np.shape[1]), np.uint8)
    for i in real_index:
          if i!=0:
              mask = labels == i
              rectangle_single[:, :][mask] = 1

    return flag_ring, rectangle_single*255


def fillhole_3(mask):
    mask = mask.astype(dtype="uint8")
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out


def main():
    net1 = GateNet_dem_512().cuda()
    net2 = GateNet_dem_512().cuda()
    net3 = GateNet_dem_512().cuda()
    net6 = MINet_Resnet().cuda()
    net7 = MSNet().cuda()
    net8 = MSNet().cuda()
    net15 = MSNet().cuda()
    net18 = MSNet().cuda()
    net23 = MSNet().cuda()
    net24 = MSNet().cuda()
    net25 = MINet_Resnet().cuda()
    net26 = MINet_Resnet().cuda()
    net29 = GateNet_dem_512().cuda()
    net30 = GateNet_dem_512().cuda()

    print ('load snapshot \'%s\' for testing' % args['snapshot1'])
    net1.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot1']+'.pth'),map_location={'cuda:1': 'cuda:1'}))
    net2.load_state_dict(
        torch.load(os.path.join(ckpt_path, exp_name, args['snapshot2'] + '.pth'), map_location={'cuda:1': 'cuda:1'}))
    net3.load_state_dict(
        torch.load(os.path.join(ckpt_path, exp_name, args['snapshot3'] + '.pth'), map_location={'cuda:1': 'cuda:1'}))
    net6.load_state_dict(
        torch.load(os.path.join(ckpt_path, exp_name, args['snapshot6'] + '.pth'), map_location={'cuda:1': 'cuda:1'}))
    net7.load_state_dict(
        torch.load(os.path.join(ckpt_path, exp_name, args['snapshot7'] + '.pth'), map_location={'cuda:1': 'cuda:1'}))
    net8.load_state_dict(
        torch.load(os.path.join(ckpt_path, exp_name, args['snapshot8'] + '.pth'), map_location={'cuda:1': 'cuda:1'}))
    net15.load_state_dict(
        torch.load(os.path.join(ckpt_path, exp_name, args['snapshot15'] + '.pth'), map_location={'cuda:1': 'cuda:1'}))
    net18.load_state_dict(
        torch.load(os.path.join(ckpt_path, exp_name, args['snapshot18'] + '.pth'), map_location={'cuda:1': 'cuda:1'}))
    net23.load_state_dict(
        torch.load(os.path.join(ckpt_path, exp_name, args['snapshot23'] + '.pth'), map_location={'cuda:1': 'cuda:1'}))
    net24.load_state_dict(
        torch.load(os.path.join(ckpt_path, exp_name, args['snapshot24'] + '.pth'), map_location={'cuda:1': 'cuda:1'}))
    net25.load_state_dict(
        torch.load(os.path.join(ckpt_path, exp_name, args['snapshot25'] + '.pth'), map_location={'cuda:1': 'cuda:1'}))
    net26.load_state_dict(
        torch.load(os.path.join(ckpt_path, exp_name, args['snapshot26'] + '.pth'), map_location={'cuda:1': 'cuda:1'}))
    net29.load_state_dict(
        torch.load(os.path.join(ckpt_path, exp_name, args['snapshot29'] + '.pth'), map_location={'cuda:1': 'cuda:1'}))
    net30.load_state_dict(
        torch.load(os.path.join(ckpt_path, exp_name, args['snapshot30'] + '.pth'), map_location={'cuda:1': 'cuda:1'}))

    net1.eval()
    net2.eval()
    net3.eval()
    net6.eval()
    net7.eval()
    net8.eval()
    net15.eval()
    net18.eval()
    net23.eval()
    net24.eval()
    net25.eval()
    net26.eval()
    net29.eval()
    net30.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            root1 = os.path.join(root,'Image')
            img_list = [os.path.splitext(f) for f in os.listdir(root1) if f.endswith('.png')]
            print(len(img_list))
            for idx, img_name in enumerate(img_list):
                print(img_name[0])
                rgb_png_path = os.path.join(root, 'Image', img_name[0] + '.png')
                img = Image.open(rgb_png_path).convert('RGB')
                w_,h_ = img.size
                img_resize = img.resize([1100,800],Image.BILINEAR)  # Foldconv cat是320
                img_resize_mask3 = img.resize([880, 640], Image.BILINEAR)  # Foldconv cat是3
                img_var = Variable(img_transform(img_resize).unsqueeze(0), volatile=True).cuda()
                img_var_mask3 = Variable(img_transform(img_resize_mask3).unsqueeze(0), volatile=True).cuda()

                mask = []
                for transformer in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()
                    rgb_trans = transformer.augment_image(img_var)
                    model_output = net1(rgb_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)
                model_output_1 = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction_1 = model_output_1.sigmoid()
                res1 = F.upsample(prediction_1, size=[h_, w_], mode='bilinear', align_corners=False)
                res1 = res1.data.cpu().numpy().squeeze()

                mask = []
                for transformer in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()
                    rgb_trans = transformer.augment_image(img_var)
                    model_output = net2(rgb_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)
                model_output_2 = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction_2 = model_output_2.sigmoid()
                res2 = F.upsample(prediction_2, size=[h_, w_], mode='bilinear', align_corners=False)
                res2 = res2.data.cpu().numpy().squeeze()

                mask = []
                for transformer in transforms_new:  # custom transforms or e.g. tta.aliases.d4_transform()
                    rgb_trans = transformer.augment_image(img_var_mask3)
                    model_output = net3(rgb_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)
                model_output_3 = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction_3 = model_output_3.sigmoid()
                res3 = F.upsample(prediction_3, size=[h_, w_], mode='bilinear', align_corners=False)
                res3 = res3.data.cpu().numpy().squeeze()

                mask = []
                for transformer in transforms_new:  # custom transforms or e.g. tta.aliases.d4_transform()
                    rgb_trans = transformer.augment_image(img_var_mask3)
                    model_output = net6(rgb_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)
                model_output_6 = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction_6 = model_output_6.sigmoid()
                res6 = F.upsample(prediction_6, size=[h_, w_], mode='bilinear', align_corners=False)
                res6 = res6.data.cpu().numpy().squeeze()

                mask = []
                for transformer in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()
                    rgb_trans = transformer.augment_image(img_var)
                    model_output = net7(rgb_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)
                model_output_7 = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction_7 = model_output_7.sigmoid()
                res7 = F.upsample(prediction_7, size=[h_, w_], mode='bilinear', align_corners=False)
                res7 = res7.data.cpu().numpy().squeeze()

                mask = []
                for transformer in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()
                    rgb_trans = transformer.augment_image(img_var)
                    model_output = net8(rgb_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)
                model_output_8 = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction_8 = model_output_8.sigmoid()
                res8 = F.upsample(prediction_8, size=[h_, w_], mode='bilinear', align_corners=False)
                res8 = res8.data.cpu().numpy().squeeze()

                mask = []
                for transformer in transforms_new:  # custom transforms or e.g. tta.aliases.d4_transform()
                    rgb_trans = transformer.augment_image(img_var_mask3)
                    model_output = net15(rgb_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)
                model_output_15 = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction_15 = model_output_15.sigmoid()
                res15 = F.upsample(prediction_15, size=[h_, w_], mode='bilinear', align_corners=False)
                res15 = res15.data.cpu().numpy().squeeze()

                mask = []
                for transformer in transforms_new:  # custom transforms or e.g. tta.aliases.d4_transform()
                    rgb_trans = transformer.augment_image(img_var_mask3)
                    model_output = net18(rgb_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)
                model_output_18 = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction_18 = model_output_18.sigmoid()
                res18 = F.upsample(prediction_18, size=[h_, w_], mode='bilinear', align_corners=False)
                res18 = res18.data.cpu().numpy().squeeze()

                mask = []
                for transformer in transforms_new:  # custom transforms or e.g. tta.aliases.d4_transform()
                    rgb_trans = transformer.augment_image(img_var_mask3)
                    model_output = net23(rgb_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)
                model_output_23 = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction_23 = model_output_23.sigmoid()
                res23 = F.upsample(prediction_23, size=[h_, w_], mode='bilinear', align_corners=False)
                res23 = res23.data.cpu().numpy().squeeze()

                mask = []
                for transformer in transforms_new:  # custom transforms or e.g. tta.aliases.d4_transform()
                    rgb_trans = transformer.augment_image(img_var_mask3)
                    model_output = net24(rgb_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)
                model_output_24 = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction_24 = model_output_24.sigmoid()
                res24 = F.upsample(prediction_24, size=[h_, w_], mode='bilinear', align_corners=False)
                res24 = res24.data.cpu().numpy().squeeze()

                mask = []
                for transformer in transforms_new:  # custom transforms or e.g. tta.aliases.d4_transform()
                    rgb_trans = transformer.augment_image(img_var_mask3)
                    model_output = net25(rgb_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)
                model_output_25 = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction_25 = model_output_25.sigmoid()
                res25 = F.upsample(prediction_25, size=[h_, w_], mode='bilinear', align_corners=False)
                res25 = res25.data.cpu().numpy().squeeze()

                mask = []
                for transformer in transforms_new:  # custom transforms or e.g. tta.aliases.d4_transform()
                    rgb_trans = transformer.augment_image(img_var_mask3)
                    model_output = net26(rgb_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)
                model_output_26 = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction_26 = model_output_26.sigmoid()
                res26 = F.upsample(prediction_26, size=[h_, w_], mode='bilinear', align_corners=False)
                res26 = res26.data.cpu().numpy().squeeze()

                mask = []
                for transformer in transforms_new:  # custom transforms or e.g. tta.aliases.d4_transform()
                    rgb_trans = transformer.augment_image(img_var_mask3)
                    model_output = net29(rgb_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)
                model_output_29 = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction_29 = model_output_29.sigmoid()
                res29 = F.upsample(prediction_29, size=[h_, w_], mode='bilinear', align_corners=False)
                res29 = res29.data.cpu().numpy().squeeze()

                mask = []
                for transformer in transforms_new:  # custom transforms or e.g. tta.aliases.d4_transform()
                    rgb_trans = transformer.augment_image(img_var_mask3)
                    model_output = net30(rgb_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)
                model_output_30 = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction_30 = model_output_30.sigmoid()
                res30 = F.upsample(prediction_30, size=[h_, w_], mode='bilinear', align_corners=False)
                res30 = res30.data.cpu().numpy().squeeze()


                res1 = 255 * res1
                res2 = 255 * res2
                res3 = 255 * res3
                res6 = 255 * res6
                res7 = 255 * res7
                res8 = 255 * res8
                res15 = 255 * res15
                res18 = 255 * res18
                res23 = 255 * res23
                res24 = 255 * res24
                res25 = 255 * res25
                res26 = 255 * res26
                res29 = 255 * res29
                res30 = 255 * res30

                res1 = (res1 + res7 + res23 + res25 + res29) / 5
                res2 = (res2 + res8 + res24 + res26 + res30) / 5
                res3 = (res3 + res6 + res15 + res18) / 4

                res1 = crf_refine(np.array(img), np.array(res1))
                res2 = crf_refine(np.array(img), np.array(res2))
                res3 = crf_refine(np.array(img), np.array(res3))

                res1[res1 > 20] = 255
                res1[res1 != 255] = 0

                res2[res2 > 20] = 255
                res2[res2 != 255] = 0

                res3[res3 > 20] = 255
                res3[res3 != 255] = 0


                _,res1 = filter_ring(res1,4)
                _,res2 = filter_ring(res2,4)
                _,res3 = filter_ring(res3,4)

                res1 = fillhole_3(res1)
                res1[res1 > 20] = 255
                res1[res1 != 255] = 0
                res2 = fillhole_3(res2)
                res2[res2 > 20] = 255
                res2[res2 != 255] = 0
                res3 = fillhole_3(res3)
                res3[res3 > 20] = 255
                res3[res3 != 255] = 0

                res2_original = res2.copy()
                res_1_2 = res1*res2
                res_1_2[res_1_2!=0] = 255
                res2[res_1_2==255] = 0
                _,res2 = filter_ring(res2,1)
                res2_original[res2==255] = 0
                res1 = res2_original + res1
                res1[res1>0] = 255
                res1[res1!= 255] = 0

                duan_index = []
                for i in range(1100):
                    if sum(res3[0:800,i])==0:
                        duan_index.append(i)
                if duan_index != []:
                    duan_index_left = duan_index[1:]
                    duan_index_left.append(0)
                    difference_x_list = list(map(lambda x: x[0]-x[1],zip(duan_index_left,duan_index)))
                    duan_location_index = [i for(i,v) in enumerate(difference_x_list)if v!=1]
                    duan_location_x = []
                    for i in range(len(duan_location_index)):
                        if i ==0:
                            start_duan = duan_index[0]
                            end_duan = duan_index[duan_location_index[i]]
                            duan_location_x.append((start_duan,end_duan))
                        else:
                            start_duan = duan_index[duan_location_index[i-1]+1]
                            end_duan = duan_index[duan_location_index[i]]
                            duan_location_x.append((start_duan, end_duan))
                    duan_location_y_res2 = []
                    for i in duan_location_x:
                        duan_res2 = np.zeros((res3.shape[0], res3.shape[1]), np.uint8)
                        duan_res2[0:800,i[0]:i[1]] = res2[0:800,i[0]:i[1]]
                        duan_res2_y_index = []
                        for j in range(800):
                            if sum(duan_res2[j, 0:1100]) != 0:
                                duan_res2_y_index.append(j)
                        if duan_res2_y_index==[]:
                            duan_res2 = np.zeros((res3.shape[0], res3.shape[1]), np.uint8)
                            duan_res2[0:800, i[0]:i[1]] = res1[0:800, i[0]:i[1]]
                            duan_res2_y_index = []
                            for i in range(800):
                                if sum(duan_res2[i, 0:1100]) != 0:
                                    duan_res2_y_index.append(i)
                        start_duan_res2_y = duan_res2_y_index[0]
                        end_duan_res2_y = duan_res2_y_index[-1]
                        duan_location_y_res2.append((start_duan_res2_y,end_duan_res2_y))
                    for i in range(len(duan_location_y_res2)):
                        if i!=len(duan_location_y_res2)-1 or duan_location_x[i][0]==0:
                            a = np.zeros((res3.shape[0], res3.shape[1]), np.uint8)
                            a[0:800,duan_location_x[i][1]:duan_location_x[i][1]+2] = res2[0:800, duan_location_x[i][1]:duan_location_x[i][1]+2]
                            b = np.zeros((res3.shape[0], res3.shape[1]), np.uint8)
                            b[0:800, duan_location_x[i][1]:duan_location_x[i][1] + 2] = res3[0:800,duan_location_x[i][1]:duan_location_x[i][1] + 2]
                            a_list = []
                            b_list = []
                            for j in range(800):
                                if sum(a[j, 0:1100]) != 0:
                                    a_list.append(j)
                                if sum(b[j, 0:1100]) != 0:
                                    b_list.append(j)
                            if a_list != []:
                                distance_res2_res3 = b_list[0]-a_list[0]
                            else:
                                distance_res2_res3 = 10

                        else:
                            a = np.zeros((res3.shape[0], res3.shape[1]), np.uint8)
                            a[0:800,duan_location_x[i][0]-2:duan_location_x[i][0]] = res2[0:800, duan_location_x[i][0]-2:duan_location_x[i][0]]
                            b = np.zeros((res3.shape[0], res3.shape[1]), np.uint8)
                            b[0:800,duan_location_x[i][0]-2:duan_location_x[i][0]] = res3[0:800, duan_location_x[i][0]-2:duan_location_x[i][0]]
                            a_list = []
                            b_list = []
                            for j in range(800):
                                if sum(a[j, 0:1100]) != 0:
                                    a_list.append(j)
                                if sum(b[j, 0:1100]) != 0:
                                    b_list.append(j)
                            if a_list != []:
                                distance_res2_res3 = b_list[0]-a_list[0]
                            else:
                                distance_res2_res3 = 10
                        res3[duan_location_y_res2[i][0]+distance_res2_res3:duan_location_y_res2[i][1]+distance_res2_res3,duan_location_x[i][0]:duan_location_x[i][1]] = res2[duan_location_y_res2[i][0]:duan_location_y_res2[i][1],duan_location_x[i][0]:duan_location_x[i][1]]

                merge_mask = 255*np.ones((res3.shape[0], res3.shape[1]), np.uint8)
                merge_mask[res1 == 255] = 0
                merge_mask[res2 == 255] = 80
                merge_mask[res3 == 255] = 160

                if args['save_results']:
                    check_mkdir(os.path.join(ckpt_path, exp_name,task,name))
                    cv2.imwrite(os.path.join(ckpt_path, exp_name ,task,name, img_name[0] + '.png'), merge_mask)



if __name__ == '__main__':
    main()
