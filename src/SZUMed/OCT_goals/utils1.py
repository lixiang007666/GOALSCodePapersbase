import os

import cv2
import numpy as np
import torch
# from medpy import metric
from PIL import Image
import scipy.ndimage as sn
from scipy import ndimage
from scipy.ndimage import zoom
from skimage import morphology
from skimage.measure import label, regionprops

import torch.nn as nn
import SimpleITK as sitk
import cv2 as cv
# from PIL import Image
import random
import scipy.signal as  ss
import time

from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import os.path as osp
import numpy as np
import os
import cv2
from skimage import morphology
import scipy
from PIL import Image
from matplotlib.pyplot import imsave
# from keras.preprocessing import image
from skimage.measure import label, regionprops
from networks.vision_transformer import SwinTnet as swinT
# from networks.vision_transformer import SwinTnetROI as swinTROI
from skimage.transform import rotate, resize
from skimage import measure, draw

from itertools import filterfalse as ifilterfalse


import matplotlib.pyplot as plt
plt.switch_backend('agg')












class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

def pixel_acc(output_od,output_oc,od_batch,oc_batch):
    metric_disc = SegmentationMetric(2)
    metric_cup = SegmentationMetric(2)

    # od_batch = od_batch.data.cpu()
    # oc_batch = oc_batch.data.cpu()
    # output_od = output_od.data.cpu()
    # output_oc = output_oc.data.cpu()

    #
    # output_od[output_od > 0.75] = 1
    # output_od[output_od <= 0.75] = 0
    #
    # output_oc[output_oc > 0.75] = 1
    # output_oc[output_oc <= 0.75] = 0


    metric_disc.addBatch(output_od.flatten(),od_batch.flatten())
    metric_cup.addBatch(output_oc.flatten(), oc_batch.flatten())

    PA_disc = metric_disc.pixelAccuracy()
    PA_cup = metric_cup.pixelAccuracy()

    iou_disc = metric_disc.meanIntersectionOverUnion()
    iou_cup = metric_cup.meanIntersectionOverUnion()

    return PA_cup, PA_disc, iou_cup, iou_disc

def vertical_diameter(binary_segmentation):
    '''
    Get the vertical diameter from a binary segmentation.
    The vertical diameter is defined as the "fattest" area of the binary_segmentation parameter.
    '''
    # get the sum of the pixels in the vertical axis
    vertical_axis_diameter = np.sum(binary_segmentation, axis=0)
    # print(vertical_axis_diameter.shape)

    # pick the maximum value
    diameter = np.max(vertical_axis_diameter, axis=0)

    # return it
    return diameter

EPS = 1e-7

def vertical_cup_to_disc_ratio(od, oc):
    '''
    Compute the vertical cup-to-disc ratio from a given labelling map.
    '''
    # compute the cup diameter

    cup_diameter = vertical_diameter(oc)
    # compute the disc diameter
    disc_diameter = vertical_diameter(od)

    return cup_diameter / (disc_diameter + EPS)


def get_vCDRs(od,oc,od_batch,oc_batch):
    # pred_od = od.cpu().numpy()
    # pred_oc = oc.cpu().numpy()
    # gt_od = od_batch.cpu().numpy()
    # gt_oc = oc_batch.cpu().numpy()
    pred_vCDR = vertical_cup_to_disc_ratio(od, oc)
    gt_vCDR = vertical_cup_to_disc_ratio(od_batch, oc_batch)
    return pred_vCDR, gt_vCDR


def compute_vCDR_error(pred_vCDR, gt_vCDR):
    '''
    Compute vCDR prediction error, along with predicted vCDR and ground truth vCDR.
    '''
    vCDR_err = np.mean(np.abs(gt_vCDR - pred_vCDR))
    return vCDR_err

def MSE_loss(pred, gt):
    d = gt - pred
    d = torch.pow(d, 2)
    d = torch.sum(d, 1)
    d = torch.sqrt(d)
    d = torch.sum(d,0)
    # print(d.shape)
    return d


def fov_loss(pred, gt):
    pdf = gt
    loss = F.l1_loss(pred, pdf, reduction='mean')
    return loss


def dice_coeff(input, target):
    '''
    in tensor fomate
    :param input:
    :param target:
    :return:
    '''
    smooth = 1.
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def dice_coeff_val(pred, target):

    pred[pred > 0.75] = 1
    pred[pred <= 0.75] = 0

    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return 1.0 - (2. * intersection) / (m1.sum() + m2.sum())


def compute_dice_coef(input, target):
    '''
    Compute dice score metric.
    '''
    input[input > 0.75] = 1
    input[input <= 0.75] = 0

    batch_size = input.shape[0]
    return sum([dice_coef_sample(input[k, :, :], target[k, :, :]) for k in range(batch_size)])


def dice_coef_sample(input, target):
    iflat = input.contiguous().view(-1)
    tflat = target.type(torch.float32).contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    res = 1.0  - (2. * intersection) / (iflat.sum() + tflat.sum())
    return res


def mean_BCE_loss(pred, gt):
    loss = F.binary_cross_entropy(pred, gt, reduction='mean')
    return loss


def get_largest_fillhole(binary):
    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=True, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    # print(C)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)

def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def isnan(x):
    return x != x

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

def test_single_volume(image, net, left_top, classes, patch_size=[256, 256],
                       test_save_path=None, case=None, z_spacing=1,split="testing"):
    # print('image label',image.shape,label.shape)

    if len(image.shape) == 4:

        B,C,H,W = image.shape

        # print(B,C,H,W)
        prediction = np.zeros((B, 4))

        # print(prediction.shape)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            slice = slice.unsqueeze(0)
            # print(slice.shape)
            input = slice
            net.eval()
            with torch.no_grad():
                # print("input.shape:",input.shape)
                # outputs1,outputs2 = net(input)

                outputs3 = net(input)

                # print(outputs1.shape, outputs3.shape)

                outputs3_od = outputs3[:, 0, :, :]
                outputs3_oc = outputs3[:, 1, :, :]
                outputs3_fov = outputs3[:, 2, :, :]

                outputs3_od = outputs3_od.cpu().detach().numpy()
                outputs3_oc = outputs3_oc.cpu().detach().numpy()
                outputs3_fov = outputs3_fov.cpu().detach().numpy()


                outputs3_od = outputs3_od.squeeze(0)
                outputs3_oc = outputs3_oc.squeeze(0)
                outputs3_fov = outputs3_fov.squeeze(0)

                outputs3_oc = zoom(outputs3_oc, (1634 / 224, 1634 / 224), order=0)
                outputs3_od = zoom(outputs3_od, (1634 / 224, 1634 / 224), order=0)
                # outputs3_fov = zoom(outputs3_fov, (1634 / 224, 1634 / 224), order=0)

                # outputs3_oc = (outputs3_oc > .75).astype(int)
                # outputs3_od = (outputs3_od > .75).astype(int)

                outputs3_oc = (outputs3_oc > .75).astype(np.uint8)
                outputs3_od = (outputs3_od > .75).astype(np.uint8)
                outputs3_fov = (outputs3_fov > .75).astype(np.uint8)
                #
                disc_mask = outputs3_od
                cup_mask = outputs3_oc
                # fovea_mask = outputs3_fov

                for i in range(5):
                    disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
                    cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
                    # fovea_mask = scipy.signal.medfilt2d(fovea_mask, 7)
                disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
                cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
                # fovea_mask = morphology.binary_erosion(fovea_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
                outputs3_od = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
                outputs3_oc = get_largest_fillhole(cup_mask).astype(np.uint8)
                # outputs3_fov = get_largest_fillhole(fovea_mask).astype(np.uint8)



    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()



    prediction = outputs3_od.copy()
    prediction[outputs3_od == 0] = 255
    prediction[outputs3_od == 1] = 128
    prediction[outputs3_oc == 1] = 0

    if test_save_path is not None:
        # prediction = prediction.squeeze(0)
        # print(image.shape,prediction.shape,label.shape)
        image = image.squeeze(0)
        # prediction = prediction[:,:,np.newaxis]
        # labelpred = labelpred.squeeze(0)
        # print(image.shape, prediction.shape, label.shape)
        # img_itk = Image.fromarray(image.astype(np.uint8))
        prd_itk = Image.fromarray(prediction.astype(np.uint8))
        # lab_itk = Image.fromarray(labelpred.astype(np.uint8))

        # img_itk.save(test_save_path + '/'+case + ".jpg")
        case = str(case[0])
        print(case)
        print( os.path.join('datasets/fovea_refuge', split, 'pred', case + '.bmp'))
        prd_itk.save(os.path.join('datasets/fovea_refuge', split, 'pred', case + '.bmp'))
        # lab_itk.save(test_save_path + '/'+ 'pred' + '/'+ case + ".png")


    # print(outputs3_fov.shape)
    outputs1 = ndimage.measurements.center_of_mass(outputs3_fov)
    outputs1 = np.array([outputs1[1], outputs1[0]])
    outputs1 = (outputs1 / 224.0) * 1634.0
    # print(outputs1)

    ### 视盘位置
    mass_centers = ndimage.measurements.center_of_mass(outputs3_od)
    mass_centers = np.array([mass_centers[1], mass_centers[0]])
    print(mass_centers)

    ### 裁剪视盘的ROI 裁剪MASK的ROI 裁剪原图的ROI
    ### 注意！！！是所有图像！！！

    # img_path = os.path.join('datasets/fovea_localization', 'testing', 'Fundus_color_images_0', case + '.jpg')
    # target_path = os.path.join('datasets/fovea_localization', 'testing', 'Fundus_color_images_plot', case + '.jpg')

    img_path1 = os.path.join('datasets/fovea_refuge', split, 'Fundus_color_images_dst', case + '.jpg')
    img_path2 = os.path.join('datasets/fovea_refuge', split, 'Disc_Cup_Masks', case + '.bmp')
    img_path3 = os.path.join('datasets/fovea_refuge', split, 'pred', case + '.bmp')
    img_path4 = os.path.join('datasets/fovea_refuge', split, 'Fundus_color_images_0', case + '.jpg')


    target_path1 = os.path.join('datasets/fovea_refuge', split, 'Fundus_color_images_ROI_dst', case + '.jpg')
    target_path2 = os.path.join('datasets/fovea_refuge', split, 'Disc_Cup_Masks_ROI', case + '.bmp')
    target_path3 = os.path.join('datasets/fovea_refuge', split, 'pred_ROI', case + '.bmp')
    target_path4 = os.path.join('datasets/fovea_refuge', split, 'Fundus_color_images_ROI', case + '.jpg')


    original = Image.open(img_path4).convert('RGB')
    Disc_cup_mask = Image.open(img_path2)
    pred_ROI = Image.open(img_path3)
    original_dst = Image.open(img_path1).convert('RGB')

    ROI_size = 448.0

    ROI_size1 = 1634.0 - ROI_size/2.0

    if mass_centers[0] > ROI_size1:
        mass_centers[0] = ROI_size1
    if mass_centers[1] > ROI_size1:
        mass_centers[1] = ROI_size1

    if mass_centers[0] <  ROI_size / 2.0:
        mass_centers[0] = ROI_size / 2.0
    if mass_centers[1]< ROI_size / 2.0:
        mass_centers[1] = ROI_size / 2.0

    leftwidth1 =  mass_centers[0] - ROI_size/2
    rightwidth1 =  mass_centers[0] +  ROI_size/2
    upperwidth1 =  mass_centers[1] -  ROI_size/2
    underwidth1 = mass_centers[1] +  ROI_size/2

    print(leftwidth1, upperwidth1, rightwidth1, underwidth1)

    ROI_write_file = os.path.join('datasets/fovea_refuge', "testing", "ROI_position.txt")

    if split == "testing":
        file = open(ROI_write_file, 'a+')

        file.write(str(leftwidth1) + '\t' +str(upperwidth1) + '\t' + str(rightwidth1) + '\t'+ str(underwidth1) +  "\n")
        file.flush()

    original = original.crop((leftwidth1, upperwidth1, rightwidth1, underwidth1))
    Disc_cup_mask = Disc_cup_mask.crop((leftwidth1, upperwidth1, rightwidth1, underwidth1))
    pred_ROI = pred_ROI.crop((leftwidth1, upperwidth1, rightwidth1, underwidth1))
    original_dst = original_dst.crop((leftwidth1, upperwidth1, rightwidth1, underwidth1))

    original.save(target_path4)
    Disc_cup_mask.save(target_path2)
    pred_ROI.save(target_path3)
    original_dst.save(target_path1)

   #### 返回中央凹的位置，视盘视杯分割结果

    return  outputs1, outputs3_od, outputs3_oc



    # img_path = os.path.join('datasets/fovea_localization', 'testing', 'Fundus_color_images_0', case + '.jpg')
    # target_path = os.path.join('datasets/fovea_localization', 'testing', 'Fundus_color_images_plot', case + '.jpg')

    # img_path = os.path.join('datasets/fovea_refuge', 'testing', 'Fundus_color_images_0', case + '.jpg')
    # target_path = os.path.join('datasets/fovea_refuge', 'testing', 'Fundus_color_images_plot', case + '.jpg')
    #
    # img_origin = cv2.imread(img_path)
    #
    # test1_x = int(outputs1[0])
    # test1_y = int(outputs1[1])
    #
    # # 画边框
    # cv2.rectangle(img_origin, (test1_x - 1, test1_y - 30), (test1_x + 1, test1_y + 30), (0, 0, 0), -1)
    # cv2.rectangle(img_origin, (test1_x - 30, test1_y - 1), (test1_x + 30, test1_y + 1), (0, 0, 0), -1)
    #
    # cv2.imwrite(target_path, img_origin)
    #
    # print("执行结束。")
    #
    #
    #
    # if test_save_path is not None:
    #     # print(image.shape, prediction.shape, label.shape)
    #     # print(prediction)
    #     with open(test_save_path + '/final_coordinate.txt', 'a', encoding='utf-8') as filetext:
    #         filetext.write(case+" "+
    #                        str(outputs1[0])+" "+str(outputs1[1])+" "+
    #                        str(outputs2[0])+" "+str(outputs2[1])+" "+
    #                        str(outputs3[0])+" "+str(outputs3[1])+"\n")
    #         filetext.close()
    #
    # return outputs1, outputs2 ,outputs3

def test_single_volumeROI(image, net, patch_size=[256, 256],
                       test_save_path=None, case=None, z_spacing=1,split="testing"):
    # print('image label',image.shape,label.shape)

    if len(image.shape) == 4:

        B,C,H,W = image.shape

        # print(B,C,H,W)
        prediction = np.zeros((B, 4))

        # print(prediction.shape)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            slice = slice.unsqueeze(0)
            # print(slice.shape)
            input = slice
            net.eval()
            with torch.no_grad():
                # print("input.shape:",input.shape)
                # outputs1,outputs2 = net(input)

                # ave_out, side_5, side_6, side_7, side_8 = net(input)

                output3_od = net(input)

                # od1 = nn.UpsamplingBilinear2d(scale_factor=2)(output3_od)
                # print(od1.shape)
                od1 = output3_od



                od = torch.argmax(torch.softmax(output3_od, dim=1), dim=1).squeeze(0)

                outputs3_od = od.cpu().detach().numpy()

                # outputs3_od = zoom(od, (448 / 224, 448 / 224), order=0)


                # # outputs3_oc = (outputs3_oc > .5).astype(np.uint8)
                # outputs3_od = (outputs3_od > .5).astype(np.uint8)
                #
                # disc_mask = outputs3_od
                # # cup_mask = outputs3_oc
                #
                # for i in range(5):
                #     disc_mask = scipy.signal.medfilt2d(disc_mask,7)
                #     # cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
                #
                # disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
                # # cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
                #
                # outputs3_od = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
                # outputs3_oc = get_largest_fillhole(cup_mask).astype(np.uint8)

    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()



    prediction = outputs3_od.copy()

    prediction[outputs3_od == 0] = 255
    prediction[outputs3_od == 1] = 128
    # prediction[outputs3_oc == 1] = 0

    if test_save_path is not None:
        # prediction = prediction.squeeze(0)
        # print(image.shape,prediction.shape,label.shape)
        image = image.squeeze(0)
        # prediction = prediction[:,:,np.newaxis]
        # labelpred = labelpred.squeeze(0)
        # print(image.shape, prediction.shape, label.shape)
        # img_itk = Image.fromarray(image.astype(np.uint8))
        prd_itk = Image.fromarray(prediction.astype(np.uint8))
        # lab_itk = Image.fromarray(labelpred.astype(np.uint8))

        # img_itk.save(test_save_path + '/'+case + ".jpg")
        case = str(case[0])
        # print(case)
        # print( os.path.join('datasets/fovea_refuge', split, 'pred_ROI', case + '.bmp'))
        prd_itk.save(os.path.join('datasets/fovea_refuge', split, 'pred_ROI', case + '.bmp'))
        # lab_itk.save(test_save_path + '/'+ 'pred' + '/'+ case + ".png")

    return od1
    # return  outputs3_od, outputs3_oc


def test_single_volume_GAMMA(config,args,dataset,image, label, net, model_file_list, left_top, patch_size=[256, 256],
                       test_save_path=None, case=None, z_spacing=1,split="testing"):
    # print('image label',image.shape,label.shape)

    if len(image.shape) == 4:

        B,C,H,W = image.shape

        # print(B,C,H,W)
        prediction = np.zeros((B, 4))

        # print(prediction.shape)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            slice = slice.unsqueeze(0)
            # print(slice.shape)
            input = slice

            with torch.no_grad():
                # print("input.shape:",input.shape)
                # outputs1,outputs2 = net(input)

                net1 = swinT(config, img_size=args.img_size, num_classes=args.num_classes).to(1)
                net1.load_state_dict(torch.load(model_file_list[0]))
                net1.to(1)
                net1.eval()

                net2 = swinT(config, img_size=args.img_size, num_classes=args.num_classes).to(1)
                net2.load_state_dict(torch.load(model_file_list[1]))
                net2.to(1)
                net2.eval()

                net3 = swinT(config, img_size=args.img_size, num_classes=args.num_classes).to(1)
                net3.load_state_dict(torch.load(model_file_list[2]))
                net3.to(1)
                net3.eval()

                net4 = swinT(config, img_size=args.img_size, num_classes=args.num_classes).to(1)
                net4.load_state_dict(torch.load(model_file_list[3]))
                net4.to(1)
                net4.eval()

                net5 = swinT(config, img_size=args.img_size, num_classes=args.num_classes).to(1)
                net5.load_state_dict(torch.load(model_file_list[4]))
                net5.to(1)
                net5.eval()


                outputs31 = net1(input)
                outputs32 = net2(input)
                outputs33 = net3(input)
                outputs34 = net4(input)
                outputs35 = net5(input)

                # print(outputs1.shape, outputs3.shape)

                outputs3_od1 = torch.sigmoid(outputs31[:, 0, :, :].to(torch.float32))
                outputs3_oc1 = torch.sigmoid(outputs31[:, 1, :, :].to(torch.float32))
                outputs3_fov1 = torch.sigmoid(outputs31[:, 2, :, :].to(torch.float32))

                outputs3_od2 = torch.sigmoid(outputs32[:, 0, :, :].to(torch.float32))
                outputs3_oc2 = torch.sigmoid(outputs32[:, 1, :, :].to(torch.float32))
                outputs3_fov2 = torch.sigmoid(outputs32[:, 2, :, :].to(torch.float32))

                outputs3_od3 = torch.sigmoid(outputs33[:, 0, :, :].to(torch.float32))
                outputs3_oc3 = torch.sigmoid(outputs33[:, 1, :, :].to(torch.float32))
                outputs3_fov3 = torch.sigmoid(outputs33[:, 2, :, :].to(torch.float32))

                outputs3_od4 = torch.sigmoid(outputs34[:, 0, :, :].to(torch.float32))
                outputs3_oc4 = torch.sigmoid(outputs34[:, 1, :, :].to(torch.float32))
                outputs3_fov4 = torch.sigmoid(outputs34[:, 2, :, :].to(torch.float32))

                outputs3_od5 = torch.sigmoid(outputs35[:, 0, :, :].to(torch.float32))
                outputs3_oc5 = torch.sigmoid(outputs35[:, 1, :, :].to(torch.float32))
                outputs3_fov5 = torch.sigmoid(outputs35[:, 2, :, :].to(torch.float32))

                outputs3_od = (outputs3_od1 + outputs3_od2 + outputs3_od3 + outputs3_od4 + outputs3_od5)/5
                outputs3_oc = (outputs3_oc1 + outputs3_oc2 + outputs3_oc3 + outputs3_oc4 + outputs3_oc5) / 5
                # outputs3_fov = (outputs3_fov1 + outputs3_fov2 + outputs3_fov3 + outputs3_fov4 + outputs3_fov5) / 5


                outputs3_od = torch.squeeze(outputs3_od)
                outputs3_oc = torch.squeeze(outputs3_oc)

                outputs3_fov1 = torch.squeeze(outputs3_fov1)
                outputs3_fov2 = torch.squeeze(outputs3_fov2)
                outputs3_fov3 = torch.squeeze(outputs3_fov3)
                outputs3_fov4 = torch.squeeze(outputs3_fov4)
                outputs3_fov5 = torch.squeeze(outputs3_fov5)

                outputs3_od = outputs3_od.cpu().detach().numpy()
                outputs3_oc = outputs3_oc.cpu().detach().numpy()

                outputs3_fov1 = outputs3_fov1.cpu().detach().numpy()
                outputs3_fov2 = outputs3_fov2.cpu().detach().numpy()
                outputs3_fov3 = outputs3_fov3.cpu().detach().numpy()
                outputs3_fov4 = outputs3_fov4.cpu().detach().numpy()
                outputs3_fov5 = outputs3_fov5.cpu().detach().numpy()


                outputs3_oc = zoom(outputs3_oc, (1934 / 224, 1934 / 224), order=0)
                outputs3_od = zoom(outputs3_od, (1934 / 224, 1934 / 224), order=0)

                # outputs3_oc = (outputs3_oc > .75).astype(int)
                # outputs3_od = (outputs3_od > .75).astype(int)

                outputs3_oc = (outputs3_oc > 0.75).astype(np.uint8)
                outputs3_od = (outputs3_od > 0.75).astype(np.uint8)


                outputs3_fov1 = (outputs3_fov1 > 0.75).astype(np.uint8)
                outputs3_fov2 = (outputs3_fov2 > 0.75).astype(np.uint8)
                outputs3_fov3 = (outputs3_fov3 > 0.75).astype(np.uint8)
                outputs3_fov4 = (outputs3_fov4 > 0.75).astype(np.uint8)
                outputs3_fov5 = (outputs3_fov5 > 0.75).astype(np.uint8)

                #
                disc_mask = outputs3_od
                cup_mask = outputs3_oc
                # fovea_mask = outputs3_fov

                for i in range(5):
                    disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
                    cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
                    # fovea_mask = scipy.signal.medfilt2d(fovea_mask, 7)
                disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
                cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
                # fovea_mask = morphology.binary_erosion(fovea_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
                outputs3_od = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
                outputs3_oc = get_largest_fillhole(cup_mask).astype(np.uint8)
                # outputs3_fov = get_largest_fillhole(fovea_mask).astype(np.uint8)



    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        # net.eval()
        # with torch.no_grad():
        #     out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        #     prediction = out.cpu().detach().numpy()

    prediction = outputs3_od.copy()
    prediction[outputs3_od == 0] = 255
    prediction[outputs3_od == 1] = 128
    prediction[outputs3_oc == 1] = 0

    prediction1 = outputs3_fov1.copy()
    prediction1[outputs3_fov1 == 0] = 255
    prediction1[outputs3_fov1 == 1] = 0
    # print(prediction)

    if left_top[0] == 529:
        offset = 529,33
        imgsize = 2992,2000
    else:
        offset = 11, 0
        imgsize = 1956, 1934

    if test_save_path is not None:
        # prediction = prediction.squeeze(0)
        # print(image.shape,prediction.shape,label.shape)
        image = image.squeeze(0)
        # prediction = prediction[:,:,np.newaxis]
        # labelpred = labelpred.squeeze(0)
        # print(image.shape, prediction.shape, label.shape)
        # img_itk = Image.fromarray(image.astype(np.uint8))
        prd_itk = Image.fromarray(prediction.astype(np.uint8))
        lab_itk = Image.fromarray(prediction1.astype(np.uint8))

        # background1 = Image.new('RGB', (imgsize[0], imgsize[1]), (255,255,255))
        # background1.paste(prd_itk, offset)
        # background2 = Image.new('RGB', (imgsize[0], imgsize[1]), (255,255,255))
        # background2.paste(lab_itk, offset)
        # img_itk.save(test_save_path + '/'+case + ".jpg")
        case = str(case[0])
        print(case)
        print( os.path.join('datasets/fovea_localization', split, 'pred', case + '.bmp'))
        prd_itk.save(os.path.join('datasets/fovea_localization', split, 'pred', case + '.bmp'))
        lab_itk.save(os.path.join('datasets/fovea_localization', split, 'pred1', case + '.bmp'))
        # lab_itk.save(test_save_path + '/'+ 'pred' + '/'+ case + ".png")


    # print(outputs3_fov.shape)
    outputs11 = ndimage.measurements.center_of_mass(outputs3_fov1)
    outputs11 = np.array([outputs11[1], outputs11[0]])
    outputs12 = ndimage.measurements.center_of_mass(outputs3_fov2)
    outputs12 = np.array([outputs12[1], outputs12[0]])
    outputs13 = ndimage.measurements.center_of_mass(outputs3_fov3)
    outputs13 = np.array([outputs13[1], outputs13[0]])
    outputs14 = ndimage.measurements.center_of_mass(outputs3_fov4)
    outputs14 = np.array([outputs14[1], outputs14[0]])
    outputs15 = ndimage.measurements.center_of_mass(outputs3_fov5)
    outputs15 = np.array([outputs15[1], outputs15[0]])
    # outputs1 = (outputs1 / 224.0) * 1934.0

    outputs1 = ((outputs11+outputs12+outputs13+outputs14+outputs15)/5.0)*(1934/224.0) + left_top

    Coordinate_write_file = os.path.join('datasets/fovea_localization', split, "pred_coordinate.txt")

    file = open(Coordinate_write_file, 'a+')
    file.write(str(outputs1[0]) + '\t' +str(outputs1[1])+ "\n")
    file.flush()


    # print(outputs1)

    ### 视盘位置
    mass_centers = ndimage.measurements.center_of_mass(outputs3_od)
    mass_centers = np.array([mass_centers[1], mass_centers[0]])
    # print(mass_centers)

    ### 裁剪视盘的ROI 裁剪MASK的ROI 裁剪原图的ROI
    ### 注意！！！是所有图像！！！

    # img_path = os.path.join('datasets/fovea_localization', 'testing', 'Fundus_color_images_0', case + '.jpg')
    # target_path = os.path.join('datasets/fovea_localization', 'testing', 'Fundus_color_images_plot', case + '.jpg')


    ##  begin
    img_path1 = os.path.join('datasets/fovea_localization', split, 'Fundus_color_images_0', case + '.jpg')
    img_path2 = os.path.join('datasets/fovea_localization', split, 'Disc_Cup_Masks', case + '.bmp')
    img_path3 = os.path.join('datasets/fovea_localization', split, 'pred', case + '.bmp')


    target_path1 = os.path.join('datasets/fovea_localization', split, 'Fundus_color_images_ROI', case + '.jpg')
    target_path2 = os.path.join('datasets/fovea_localization', split, 'Disc_Cup_Masks_ROI', case + '.bmp')
    target_path3 = os.path.join('datasets/fovea_localization', split, 'pred_ROI', case + '.bmp')

    original = Image.open(img_path1).convert('RGB')
    Disc_cup_mask = Image.open(img_path2)
    pred_ROI = Image.open(img_path3)

    ROI_size = 448.0

    ROI_size1 = 1934.0 - ROI_size/2.0

    if mass_centers[0] > ROI_size1:
        mass_centers[0] = ROI_size1
    if mass_centers[1] > ROI_size1:
        mass_centers[1] = ROI_size1

    if mass_centers[0] <  ROI_size / 2.0:
        mass_centers[0] = ROI_size / 2.0
    if mass_centers[1]< ROI_size / 2.0:
        mass_centers[1] = ROI_size / 2.0

    leftwidth1 =  mass_centers[0] - ROI_size/2
    rightwidth1 =  mass_centers[0] +  ROI_size/2
    upperwidth1 =  mass_centers[1] -  ROI_size/2
    underwidth1 = mass_centers[1] +  ROI_size/2


    ROI_write_file = os.path.join('datasets/fovea_localization', "testing", "ROI_position.txt.txt")

    if split == "testing":
        file = open(ROI_write_file, 'a+')
        file.write(str(leftwidth1+offset[0]) + '\t' +str(upperwidth1+offset[1]) + "\n")
        file.flush()

    original = original.crop((leftwidth1+offset[0], upperwidth1+offset[1], rightwidth1+offset[0], underwidth1+offset[1]))
    Disc_cup_mask = Disc_cup_mask.crop((leftwidth1, upperwidth1, rightwidth1, underwidth1))
    pred_ROI = pred_ROI.crop((leftwidth1, upperwidth1, rightwidth1, underwidth1))
    original.save(target_path1)
    Disc_cup_mask.save(target_path2)
    pred_ROI.save(target_path3)

   #### 返回中央凹的位置，视盘视杯分割结果

    return  outputs1, outputs3_od, outputs3_oc



    # img_path = os.path.join('datasets/fovea_localization', 'testing', 'Fundus_color_images_0', case + '.jpg')
    # target_path = os.path.join('datasets/fovea_localization', 'testing', 'Fundus_color_images_plot', case + '.jpg')

    # img_path = os.path.join('datasets/fovea_refuge', 'testing', 'Fundus_color_images_0', case + '.jpg')
    # target_path = os.path.join('datasets/fovea_refuge', 'testing', 'Fundus_color_images_plot', case + '.jpg')
    #
    # img_origin = cv2.imread(img_path)
    #
    # test1_x = int(outputs1[0])
    # test1_y = int(outputs1[1])
    #
    # # 画边框
    # cv2.rectangle(img_origin, (test1_x - 1, test1_y - 30), (test1_x + 1, test1_y + 30), (0, 0, 0), -1)
    # cv2.rectangle(img_origin, (test1_x - 30, test1_y - 1), (test1_x + 30, test1_y + 1), (0, 0, 0), -1)
    #
    # cv2.imwrite(target_path, img_origin)
    #
    # print("执行结束。")
    #
    #
    #
    # if test_save_path is not None:
    #     # print(image.shape, prediction.shape, label.shape)
    #     # print(prediction)
    #     with open(test_save_path + '/final_coordinate.txt', 'a', encoding='utf-8') as filetext:
    #         filetext.write(case+" "+
    #                        str(outputs1[0])+" "+str(outputs1[1])+" "+
    #                        str(outputs2[0])+" "+str(outputs2[1])+" "+
    #                        str(outputs3[0])+" "+str(outputs3[1])+"\n")
    #         filetext.close()
    #
    # return outputs1, outputs2 ,outputs3

def test_single_volumeROI_GAMMA(config,args,dataset,image, net, model_file_list,left_top, patch_size=[256, 256],
                       test_save_path=None, case=None, z_spacing=1,split="testing"):
    # print('image label',image.shape,label.shape)

    if len(image.shape) == 4:

        B,C,H,W = image.shape

        # print(B,C,H,W)
        prediction = np.zeros((B, 4))

        # print(prediction.shape)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            slice = slice.unsqueeze(0)
            # print(slice.shape)
            input = slice

            with torch.no_grad():
                # print("input.shape:",input.shape)
                # outputs1,outputs2 = net(input)

                net1 = swinTROI(config, img_size=args.img_size, num_classes=args.num_classes).to(1)
                net1.load_state_dict(torch.load(model_file_list[0]))
                net1.to(1)
                net1.eval()

                net2 = swinTROI(config, img_size=args.img_size, num_classes=args.num_classes).to(1)
                net2.load_state_dict(torch.load(model_file_list[1]))
                net2.to(1)
                net2.eval()

                net3 = swinTROI(config, img_size=args.img_size, num_classes=args.num_classes).to(1)
                net3.load_state_dict(torch.load(model_file_list[2]))
                net3.to(1)
                net3.eval()

                net4 = swinTROI(config, img_size=args.img_size, num_classes=args.num_classes).to(1)
                net4.load_state_dict(torch.load(model_file_list[3]))
                net4.to(1)
                net4.eval()

                net5 = swinTROI(config, img_size=args.img_size, num_classes=args.num_classes).to(1)
                net5.load_state_dict(torch.load(model_file_list[4]))
                net5.to(1)
                net5.eval()


                outputs31 = net1(input)
                outputs32 = net2(input)
                outputs33 = net3(input)
                outputs34 = net4(input)
                outputs35 = net5(input)

                # print(outputs1.shape, outputs3.shape)

                outputs3_od1 = torch.sigmoid(outputs31[:, 0, :, :].to(torch.float32))
                outputs3_oc1 = torch.sigmoid(outputs31[:, 1, :, :].to(torch.float32))


                outputs3_od2 = torch.sigmoid(outputs32[:, 0, :, :].to(torch.float32))
                outputs3_oc2 = torch.sigmoid(outputs32[:, 1, :, :].to(torch.float32))


                outputs3_od3 = torch.sigmoid(outputs33[:, 0, :, :].to(torch.float32))
                outputs3_oc3 = torch.sigmoid(outputs33[:, 1, :, :].to(torch.float32))


                outputs3_od4 = torch.sigmoid(outputs34[:, 0, :, :].to(torch.float32))
                outputs3_oc4 = torch.sigmoid(outputs34[:, 1, :, :].to(torch.float32))


                outputs3_od5 = torch.sigmoid(outputs35[:, 0, :, :].to(torch.float32))
                outputs3_oc5 = torch.sigmoid(outputs35[:, 1, :, :].to(torch.float32))


                outputs3_od = (outputs3_od1 + outputs3_od2 + outputs3_od3 + outputs3_od4 + outputs3_od5)/5
                outputs3_oc = (outputs3_oc1 + outputs3_oc2 + outputs3_oc3 + outputs3_oc4 + outputs3_oc5) / 5
                # outputs3_fov = (outputs3_fov1 + outputs3_fov2 + outputs3_fov3 + outputs3_fov4 + outputs3_fov5) / 5


                outputs3_od = torch.squeeze(outputs3_od)
                outputs3_oc = torch.squeeze(outputs3_oc)



                outputs3_od = outputs3_od.cpu().detach().numpy()
                outputs3_oc = outputs3_oc.cpu().detach().numpy()



                outputs3_oc = zoom(outputs3_oc, (448 / 224, 448 / 224), order=0)
                outputs3_od = zoom(outputs3_od, (448 / 224, 448 / 224), order=0)

                # outputs3_oc = (outputs3_oc > .75).astype(int)
                # outputs3_od = (outputs3_od > .75).astype(int)

                outputs3_oc = (outputs3_oc > 0.5).astype(np.uint8)
                outputs3_od = (outputs3_od > 0.5).astype(np.uint8)

                #
                # disc_mask = outputs3_od
                # cup_mask = outputs3_oc
                # # fovea_mask = outputs3_fov
                #
                # for i in range(5):
                #     disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
                #     cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
                #     # fovea_mask = scipy.signal.medfilt2d(fovea_mask, 7)
                # disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
                # cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
                # # fovea_mask = morphology.binary_erosion(fovea_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
                # outputs3_od = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
                # outputs3_oc = get_largest_fillhole(cup_mask).astype(np.uint8)
                # # outputs3_fov = get_largest_fillhole(fovea_mask).astype(np.uint8)



    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        # net.eval()
        # with torch.no_grad():
        #     out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        #     prediction = out.cpu().detach().numpy()

    prediction = outputs3_od.copy()
    prediction[outputs3_od == 0] = 255
    prediction[outputs3_od == 1] = 128
    prediction[outputs3_oc == 1] = 0
    print(case)
    case = str(case[0])
    print(case)
    img_path1 = os.path.join('datasets/fovea_localization', split, 'Fundus_color_images_0', case + '.jpg')


    original = Image.open(img_path1).convert('RGB')

    base_width, base_height = original.size

    offset = left_top[0], left_top[1]
    imgsize = base_width, base_height

    if test_save_path  is not None and split == 'testing' :
        # prediction = prediction.squeeze(0)
        # print(image.shape,prediction.shape,label.shape)
        image = image.squeeze(0)
        # prediction = prediction[:,:,np.newaxis]
        # labelpred = labelpred.squeeze(0)
        # print(image.shape, prediction.shape, label.shape)
        # img_itk = Image.fromarray(image.astype(np.uint8))
        prd_itk = Image.fromarray(prediction.astype(np.uint8))

        background1 = Image.new('RGB', (imgsize[0], imgsize[1]), (255,255,255))
        background1.paste(prd_itk, offset)
        # background2 = Image.new('RGB', (imgsize[0], imgsize[1]), (255,255,255))
        # background2.paste(lab_itk, offset)
        # img_itk.save(test_save_path + '/'+case + ".jpg")
        print( os.path.join('datasets/fovea_localization', split, 'ROI_pred', case + '.bmp'))
        background1.save(os.path.join('datasets/fovea_localization', split, 'ROI_pred', case + '.bmp'))
        # lab_itk.save(test_save_path + '/'+ 'pred' + '/'+ case + ".png")

   #### 返回中央凹的位置，视盘视杯分割结果

    return  outputs3_od, outputs3_oc












def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)

