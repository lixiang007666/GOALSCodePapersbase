#!/usr/bin/python3
#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.FoldConv import ASPP,FoldConv_aspp_with_batch
import torch
import timm

class FPN_aspp(nn.Module):


    def __init__(self):
        super(FPN_aspp, self).__init__()
        # self.bkbone = timm.create_model('resnet50d', features_only=True, pretrained=True)
        self.bkbone = timm.create_model('resnext101_32x8d', features_only=True, pretrained=True)
        ###############################Transition Layer########################################
        self.dem1 = FoldConv_aspp_with_batch(in_channel=2048,
                      out_channel=64,
                      out_size=384 // 16,
                      kernel_size=3,
                      stride=1,
                      padding=2,
                      dilation=2,
                      win_size=2,
                      win_padding=0,
        )
        self.dem1_bn = nn.BatchNorm2d(64)
        # self.dem1 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem2 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem3 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem4 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        ################################FPN branch#######################################
        self.output1_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output4_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output5_1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def forward(self, x):
        input = x
        B, _, _, _ = input.size()
        E1, E2, E3, E4, E5 = self.bkbone(x)
        ################################Transition Layer#######################################
        T5 = F.relu(self.dem1_bn(self.dem1(E5)))
        T4 = self.dem2(E4)
        T3 = self.dem3(E3)
        T2 = self.dem4(E2)
        T1 = self.dem5(E1)
        ################################Decoder Layer#######################################
        D4_1 = self.output2_1(F.upsample(T5, size=E4.size()[2:], mode='bilinear')+T4)
        D3_1 = self.output3_1(F.upsample(D4_1, size=E3.size()[2:], mode='bilinear')+T3)
        D2_1 = self.output4_1(F.upsample(D3_1, size=E2.size()[2:], mode='bilinear')+T2)
        D1_1 = self.output5_1(F.upsample(D2_1, size=E1.size()[2:], mode='bilinear')+T1)


        ################################Gated Parallel&Dual branch residual fuse#######################################
        output_fpn_p = F.upsample(D1_1, size=input.size()[2:], mode='bilinear')
        # output_res = self.out_res(torch.cat((D1,F.upsample(T5,size=E1.size()[2:], mode='bilinear'),F.upsample(T4,size=E1.size()[2:], mode='bilinear'),F.upsample(T3,size=E1.size()[2:], mode='bilinear'),F.upsample(T2,size=E1.size()[2:], mode='bilinear'),F.upsample(T1,size=E1.size()[2:], mode='bilinear')),1))
        # output_res =  F.upsample(output_res,size=input.size()[2:], mode='bilinear')
        # pre_sal = output_fpn + output_res
        #######################################################################
        if self.training:
            # return output_fpn, pre_sal
            return output_fpn_p
        # return F.sigmoid(pre_sal)
        return output_fpn_p


class FPN_two_decoder(nn.Module):


    def __init__(self):
        super(FPN_two_decoder, self).__init__()
        self.bkbone = timm.create_model('resnet50d', features_only=True, pretrained=True)
        ###############################Transition Layer########################################
        self.dem1 = ASPP(2048,64)
        # self.dem1 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem2 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem3 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem4 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        ################################FPN branch#######################################
        self.output1_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output4_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output5_1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

        self.output1_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output4_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output5_2 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def forward(self, x):
        input = x
        B, _, _, _ = input.size()
        E1, E2, E3, E4, E5 = self.bkbone(x)
        ################################Transition Layer#######################################
        T5 = self.dem1(E5)
        T4 = self.dem2(E4)
        T3 = self.dem3(E3)
        T2 = self.dem4(E2)
        T1 = self.dem5(E1)
        ################################Decoder Layer#######################################
        D4_1 = self.output2_1(F.upsample(T5, size=E4.size()[2:], mode='bilinear')+T4)
        D3_1 = self.output3_1(F.upsample(D4_1, size=E3.size()[2:], mode='bilinear')+T3)
        D2_1 = self.output4_1(F.upsample(D3_1, size=E2.size()[2:], mode='bilinear')+T2)
        D1_1 = self.output5_1(F.upsample(D2_1, size=E1.size()[2:], mode='bilinear')+T1)

        D4_2 = self.output2_2(F.upsample(T5, size=E4.size()[2:], mode='bilinear')+T4)
        D3_2 = self.output3_2(F.upsample(D4_2, size=E3.size()[2:], mode='bilinear')+T3)
        D2_2 = self.output4_2(F.upsample(D3_2, size=E2.size()[2:], mode='bilinear')+T2)
        D1_2 = self.output5_2(F.upsample(D2_2, size=E1.size()[2:], mode='bilinear')+T1)

        ################################Gated Parallel&Dual branch residual fuse#######################################
        output_fpn_hl = F.upsample(D1_1, size=input.size()[2:], mode='bilinear')
        output_fpn_p = F.upsample(D1_2, size=input.size()[2:], mode='bilinear')
        # output_res = self.out_res(torch.cat((D1,F.upsample(T5,size=E1.size()[2:], mode='bilinear'),F.upsample(T4,size=E1.size()[2:], mode='bilinear'),F.upsample(T3,size=E1.size()[2:], mode='bilinear'),F.upsample(T2,size=E1.size()[2:], mode='bilinear'),F.upsample(T1,size=E1.size()[2:], mode='bilinear')),1))
        # output_res =  F.upsample(output_res,size=input.size()[2:], mode='bilinear')
        # pre_sal = output_fpn + output_res
        #######################################################################
        if self.training:
            # return output_fpn, pre_sal
            return output_fpn_hl,output_fpn_p
        # return F.sigmoid(pre_sal)
        return output_fpn_hl,output_fpn_p


class FPN_three_decoder(nn.Module):


    def __init__(self):
        super(FPN_three_decoder, self).__init__()
        self.bkbone = timm.create_model('resnet50d', features_only=True, pretrained=True)
        ###############################Transition Layer########################################
        self.dem1 = ASPP(2048,64)
        # self.dem1 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem2 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem3 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem4 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        ################################FPN branch#######################################
        self.output1_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output4_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output5_1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

        self.output1_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output4_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output5_2 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

        self.output1_3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2_3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3_3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output4_3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output5_3 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def forward(self, x):
        input = x
        B, _, _, _ = input.size()
        E1, E2, E3, E4, E5 = self.bkbone(x)
        ################################Transition Layer#######################################
        T5 = self.dem1(E5)
        T4 = self.dem2(E4)
        T3 = self.dem3(E3)
        T2 = self.dem4(E2)
        T1 = self.dem5(E1)
        ################################Decoder Layer#######################################
        D4_1 = self.output2_1(F.upsample(T5, size=E4.size()[2:], mode='bilinear')+T4)
        D3_1 = self.output3_1(F.upsample(D4_1, size=E3.size()[2:], mode='bilinear')+T3)
        D2_1 = self.output4_1(F.upsample(D3_1, size=E2.size()[2:], mode='bilinear')+T2)
        D1_1 = self.output5_1(F.upsample(D2_1, size=E1.size()[2:], mode='bilinear')+T1)

        D4_2 = self.output2_2(F.upsample(T5, size=E4.size()[2:], mode='bilinear')+T4)
        D3_2 = self.output3_2(F.upsample(D4_2, size=E3.size()[2:], mode='bilinear')+T3)
        D2_2 = self.output4_2(F.upsample(D3_2, size=E2.size()[2:], mode='bilinear')+T2)
        D1_2 = self.output5_2(F.upsample(D2_2, size=E1.size()[2:], mode='bilinear')+T1)

        D4_3 = self.output2_3(F.upsample(T5, size=E4.size()[2:], mode='bilinear')+T4)
        D3_3 = self.output3_3(F.upsample(D4_3, size=E3.size()[2:], mode='bilinear')+T3)
        D2_3 = self.output4_3(F.upsample(D3_3, size=E2.size()[2:], mode='bilinear')+T2)
        D1_3 = self.output5_3(F.upsample(D2_3, size=E1.size()[2:], mode='bilinear')+T1)

        ################################Gated Parallel&Dual branch residual fuse#######################################
        output_fpn_ring = F.upsample(D1_1, size=input.size()[2:], mode='bilinear')
        output_fpn_hl = F.upsample(D1_2, size=input.size()[2:], mode='bilinear')
        output_fpn_p = F.upsample(D1_3, size=input.size()[2:], mode='bilinear')
        # output_res = self.out_res(torch.cat((D1,F.upsample(T5,size=E1.size()[2:], mode='bilinear'),F.upsample(T4,size=E1.size()[2:], mode='bilinear'),F.upsample(T3,size=E1.size()[2:], mode='bilinear'),F.upsample(T2,size=E1.size()[2:], mode='bilinear'),F.upsample(T1,size=E1.size()[2:], mode='bilinear')),1))
        # output_res =  F.upsample(output_res,size=input.size()[2:], mode='bilinear')
        # pre_sal = output_fpn + output_res
        #######################################################################
        if self.training:
            # return output_fpn, pre_sal
            return output_fpn_ring,output_fpn_hl,output_fpn_p
        # return F.sigmoid(pre_sal)
        return output_fpn_ring,output_fpn_hl,output_fpn_p


class FPN_two_decoder_deepsupervision(nn.Module):


    def __init__(self):
        super(FPN_two_decoder_deepsupervision, self).__init__()
        self.bkbone = timm.create_model('resnet50d', features_only=True, pretrained=True)
        ###############################Transition Layer########################################
        self.dem1 = ASPP(2048,64)
        # self.dem1 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem2 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem3 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem4 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        ################################FPN branch#######################################
        self.output1_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output4_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output5_1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

        self.sideout_4_1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))
        self.sideout_3_1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))
        self.sideout_2_1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

        self.sideout_4_2 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))
        self.sideout_3_2 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))
        self.sideout_2_2 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

        self.output1_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output4_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output5_2 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def forward(self, x):
        input = x
        B, _, _, _ = input.size()
        E1, E2, E3, E4, E5 = self.bkbone(x)
        ################################Transition Layer#######################################
        T5 = self.dem1(E5)
        T4 = self.dem2(E4)
        T3 = self.dem3(E3)
        T2 = self.dem4(E2)
        T1 = self.dem5(E1)
        ################################Decoder Layer#######################################
        D4_1 = self.output2_1(F.upsample(T5, size=E4.size()[2:], mode='bilinear')+T4)
        sideout_4_1 = self.sideout_4_1(D4_1)
        D3_1 = self.output3_1(F.upsample(D4_1, size=E3.size()[2:], mode='bilinear')+T3)
        sideout_3_1 = self.sideout_3_1(D3_1)
        D2_1 = self.output4_1(F.upsample(D3_1, size=E2.size()[2:], mode='bilinear')+T2)
        sideout_2_1 = self.sideout_2_1(D2_1)
        D1_1 = self.output5_1(F.upsample(D2_1, size=E1.size()[2:], mode='bilinear')+T1)


        D4_2 = self.output2_2(F.upsample(T5, size=E4.size()[2:], mode='bilinear')+T4)
        sideout_4_2 = self.sideout_4_2(D4_2)
        D3_2 = self.output3_2(F.upsample(D4_2, size=E3.size()[2:], mode='bilinear')+T3)
        sideout_3_2 = self.sideout_3_2(D3_2)
        D2_2 = self.output4_2(F.upsample(D3_2, size=E2.size()[2:], mode='bilinear')+T2)
        sideout_2_2 = self.sideout_2_2(D2_2)
        D1_2 = self.output5_2(F.upsample(D2_2, size=E1.size()[2:], mode='bilinear')+T1)

        ################################Gated Parallel&Dual branch residual fuse#######################################
        output_fpn_hl = F.upsample(D1_1, size=input.size()[2:], mode='bilinear')
        output_fpn_p = F.upsample(D1_2, size=input.size()[2:], mode='bilinear')
        # output_res = self.out_res(torch.cat((D1,F.upsample(T5,size=E1.size()[2:], mode='bilinear'),F.upsample(T4,size=E1.size()[2:], mode='bilinear'),F.upsample(T3,size=E1.size()[2:], mode='bilinear'),F.upsample(T2,size=E1.size()[2:], mode='bilinear'),F.upsample(T1,size=E1.size()[2:], mode='bilinear')),1))
        # output_res =  F.upsample(output_res,size=input.size()[2:], mode='bilinear')
        # pre_sal = output_fpn + output_res
        #######################################################################
        if self.training:
            # return output_fpn, pre_sal
            return sideout_4_1, sideout_3_1, sideout_2_1,output_fpn_hl,sideout_4_2,sideout_3_2,sideout_2_2,output_fpn_p
        # return F.sigmoid(pre_sal)
        return output_fpn_hl,output_fpn_p


class FPN_two_decoder_attention(nn.Module):


    def __init__(self):
        super(FPN_two_decoder_attention, self).__init__()
        self.bkbone = timm.create_model('resnet50d', features_only=True, pretrained=True)
        ###############################Transition Layer########################################
        self.dem1 = ASPP(2048,64)
        self.dem2 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem3 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem4 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        ################################FPN branch#######################################
        self.output1_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output4_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output5_1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

        self.output1_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output4_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output5_2 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def forward(self, x):
        input = x
        B, _, _, _ = input.size()
        E1, E2, E3, E4, E5 = self.bkbone(x)
        ################################Transition Layer#######################################
        T5 = self.dem1(E5)
        T4 = self.dem2(E4)
        T3 = self.dem3(E3)
        T2 = self.dem4(E2)
        T1 = self.dem5(E1)
        ################################Decoder Layer#######################################
        D4_1 = self.output2_1(F.upsample(T5, size=E4.size()[2:], mode='bilinear')+T4)
        D3_1 = self.output3_1(F.upsample(D4_1, size=E3.size()[2:], mode='bilinear')+T3)
        D2_1 = self.output4_1(F.upsample(D3_1, size=E2.size()[2:], mode='bilinear')+T2)
        D1_1 = self.output5_1(F.upsample(D2_1, size=E1.size()[2:], mode='bilinear')+T1)

        D4_2 = self.output2_2(F.upsample(T5, size=E4.size()[2:], mode='bilinear')*F.upsample(F.sigmoid(D1_1), size=E4.size()[2:], mode='bilinear')+T4)
        D3_2 = self.output3_2(F.upsample(D4_2, size=E3.size()[2:], mode='bilinear')*F.upsample(F.sigmoid(D1_1), size=E3.size()[2:], mode='bilinear')+T3)
        D2_2 = self.output4_2(F.upsample(D3_2, size=E2.size()[2:], mode='bilinear')*F.upsample(F.sigmoid(D1_1), size=E2.size()[2:], mode='bilinear')+T2)
        D1_2 = self.output5_2(F.upsample(D2_2, size=E1.size()[2:], mode='bilinear')*F.upsample(F.sigmoid(D1_1), size=E1.size()[2:], mode='bilinear')+T1)

        ################################Gated Parallel&Dual branch residual fuse#######################################
        output_fpn_hl = F.upsample(D1_1, size=input.size()[2:], mode='bilinear')
        output_fpn_p = F.upsample(D1_2, size=input.size()[2:], mode='bilinear')
        # output_res = self.out_res(torch.cat((D1,F.upsample(T5,size=E1.size()[2:], mode='bilinear'),F.upsample(T4,size=E1.size()[2:], mode='bilinear'),F.upsample(T3,size=E1.size()[2:], mode='bilinear'),F.upsample(T2,size=E1.size()[2:], mode='bilinear'),F.upsample(T1,size=E1.size()[2:], mode='bilinear')),1))
        # output_res =  F.upsample(output_res,size=input.size()[2:], mode='bilinear')
        # pre_sal = output_fpn + output_res
        #######################################################################
        if self.training:
            # return output_fpn, pre_sal
            return output_fpn_hl,output_fpn_p
        # return F.sigmoid(pre_sal)
        return output_fpn_hl,output_fpn_p


class FPN_one_decoder_two_channel(nn.Module):


    def __init__(self):
        super(FPN_one_decoder_two_channel, self).__init__()
        self.bkbone = timm.create_model('resnet50d', features_only=True, pretrained=True)
        ###############################Transition Layer########################################
        self.dem1 = ASPP(2048,64)
        self.dem2 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem3 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem4 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        ################################FPN branch#######################################
        self.output1_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output4_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output5_1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def forward(self, x):
        input = x
        B, _, _, _ = input.size()
        E1, E2, E3, E4, E5 = self.bkbone(x)
        ################################Transition Layer#######################################
        T5 = self.dem1(E5)
        T4 = self.dem2(E4)
        T3 = self.dem3(E3)
        T2 = self.dem4(E2)
        T1 = self.dem5(E1)
        ################################Decoder Layer#######################################
        D4_1 = self.output2_1(F.upsample(T5, size=E4.size()[2:], mode='bilinear')+T4)
        D3_1 = self.output3_1(F.upsample(D4_1, size=E3.size()[2:], mode='bilinear')+T3)
        D2_1 = self.output4_1(F.upsample(D3_1, size=E2.size()[2:], mode='bilinear')+T2)
        D1_1 = self.output5_1(F.upsample(D2_1, size=E1.size()[2:], mode='bilinear')+T1)

        ################################Gated Parallel&Dual branch residual fuse#######################################
        output_fpn = F.upsample(D1_1, size=input.size()[2:], mode='bilinear')
        # output_res = self.out_res(torch.cat((D1,F.upsample(T5,size=E1.size()[2:], mode='bilinear'),F.upsample(T4,size=E1.size()[2:], mode='bilinear'),F.upsample(T3,size=E1.size()[2:], mode='bilinear'),F.upsample(T2,size=E1.size()[2:], mode='bilinear'),F.upsample(T1,size=E1.size()[2:], mode='bilinear')),1))
        # output_res =  F.upsample(output_res,size=input.size()[2:], mode='bilinear')
        # pre_sal = output_fpn + output_res
        #######################################################################
        if self.training:
            # return output_fpn, pre_sal
            return output_fpn
        # return F.sigmoid(pre_sal)
        return output_fpn

if __name__ == "__main__":
    model = FPN().eval()
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
