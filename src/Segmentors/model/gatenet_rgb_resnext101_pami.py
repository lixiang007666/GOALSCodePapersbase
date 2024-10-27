#!/usr/bin/python3
#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.FoldConv import FoldConv_aspp
import torch
import timm
# from torchstat import stat
# from thop import clever_format
# from thop import profile

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            # m.initialize()
            pass


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)




class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1_MAX = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1_MAX)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('./saved_model/resnet50-19c8e357.pth'), strict=False)


class GateNet(nn.Module):


    def __init__(self):
        super(GateNet, self).__init__()
        # self.bkbone = ResNet()
        self.bkbone = timm.create_model('resnext101_32x8d', features_only=True, pretrained=True)
        ################################Gate#######################################
        self.attention_feature5 = nn.Sequential(nn.Conv2d(64+64, 2, kernel_size=3, padding=1))
        self.attention_feature4 = nn.Sequential(nn.Conv2d(64+64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature3 = nn.Sequential(nn.Conv2d(64+64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature2 = nn.Sequential(nn.Conv2d(64+64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature1 = nn.Sequential(nn.Conv2d(64+64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                                 nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                                 nn.Conv2d(64, 2, kernel_size=3, padding=1))
        ###############################Transition Layer########################################
        # self.dem1 = nn.Sequential(FoldConv_aspp(in_channel=2048,
        #               out_channel=512,
        #               out_size=384 // 16,
        #               kernel_size=3,
        #               stride=1,
        #               padding=2,
        #               dilation=2,
        #               win_size=2,
        #               win_padding=0,
        # ), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.dem1 = FoldConv_aspp(in_channel=2048,
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
        self.dem2 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem3 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem4 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.T54321_5 = nn.Sequential(nn.Conv2d(64*5, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.T54321_4 = nn.Sequential(nn.Conv2d(64*5, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.T54321_3 = nn.Sequential(nn.Conv2d(64*5, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.T54321_2 = nn.Sequential(nn.Conv2d(64*5, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.T54321_1 = nn.Sequential(nn.Conv2d(64*5, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))

        ################################FPN branch#######################################
        self.output1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output5 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))
        ################################Parallel branch#######################################
        self.out_res = nn.Sequential(nn.Conv2d(64+64+64+64+64+1, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 1, kernel_size=3, padding=1))
        # self.initialize()
        # self.bkbone.initialize()
        #######################################################################
        # for m in self.modules():
        #     if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
        #         m.inplace = True


    def forward(self, x):
        input = x
        B, _, _, _ = input.size()
        E1, E2, E3, E4, E5 = self.bkbone(x)
        ################################Transition Layer#######################################
        # print(E5.shape)
        T5 = F.relu(self.dem1_bn(self.dem1(E5)))
        T4 = self.dem2(E4)
        T3 = self.dem3(E3)
        T2 = self.dem4(E2)
        T1 = self.dem5(E1)

        T54321_5 = self.T54321_5(torch.cat((T5,F.upsample(T4, size=T5.size()[2:], mode='bilinear'),F.upsample(T3, size=T5.size()[2:], mode='bilinear'),F.upsample(T2, size=T5.size()[2:], mode='bilinear'),F.upsample(T1, size=T5.size()[2:], mode='bilinear')),1))
        T54321_4 = self.T54321_4(torch.cat((T4,F.upsample(T5, size=T4.size()[2:], mode='bilinear'),F.upsample(T3, size=T4.size()[2:], mode='bilinear'),F.upsample(T2, size=T4.size()[2:], mode='bilinear'),F.upsample(T1, size=T4.size()[2:], mode='bilinear')),1))
        T54321_3 = self.T54321_3(torch.cat((T3,F.upsample(T5, size=T3.size()[2:], mode='bilinear'),F.upsample(T4, size=T3.size()[2:], mode='bilinear'),F.upsample(T2, size=T3.size()[2:], mode='bilinear'),F.upsample(T1, size=T3.size()[2:], mode='bilinear')),1))
        T54321_2 = self.T54321_2(torch.cat((T2,F.upsample(T5, size=T2.size()[2:], mode='bilinear'),F.upsample(T4, size=T2.size()[2:], mode='bilinear'),F.upsample(T3, size=T2.size()[2:], mode='bilinear'),F.upsample(T1, size=T2.size()[2:], mode='bilinear')),1))
        T54321_1 = self.T54321_1(torch.cat((T1,F.upsample(T5, size=T1.size()[2:], mode='bilinear'),F.upsample(T4, size=T1.size()[2:], mode='bilinear'),F.upsample(T3, size=T1.size()[2:], mode='bilinear'),F.upsample(T2, size=T1.size()[2:], mode='bilinear')),1))
        ################################Gated FPN#######################################
        G5 = self.attention_feature1(torch.cat((T54321_5, T5), 1))
        G5 = F.adaptive_avg_pool2d(F.sigmoid(G5), 1)
        D5 = self.output1(G5[:, 0, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * T5)

        G4 = self.attention_feature2(torch.cat((T54321_4,F.upsample(D5, size=T54321_4.size()[2:], mode='bilinear')),1))
        G4 = F.adaptive_avg_pool2d(F.sigmoid(G4),1)
        D4 = self.output2(F.upsample(D5, size=E4.size()[2:], mode='bilinear')+G4[:, 0,:,:].unsqueeze(1).repeat(1,64,1,1)*T4)

        G3 = self.attention_feature3(torch.cat((T54321_3,F.upsample(D4, size=T54321_3.size()[2:], mode='bilinear')),1))
        G3 = F.adaptive_avg_pool2d(F.sigmoid(G3),1)
        D3 = self.output3(F.upsample(D4, size=E3.size()[2:], mode='bilinear')+G3[:, 0,:,:].unsqueeze(1).repeat(1,64,1,1)*T3)

        G2 = self.attention_feature4(torch.cat((T54321_2,F.upsample(D3, size=T54321_2.size()[2:], mode='bilinear')),1))
        G2 = F.adaptive_avg_pool2d(F.sigmoid(G2),1)
        D2 = self.output4(F.upsample(D3, size=E2.size()[2:], mode='bilinear')+G2[:, 0,:,:].unsqueeze(1).repeat(1,64,1,1)*T2)

        G1 = self.attention_feature5(torch.cat((T54321_1,F.upsample(D2, size=T54321_1.size()[2:], mode='bilinear')),1))
        G1 = F.adaptive_avg_pool2d(F.sigmoid(G1),1)
        D1 = self.output5(F.upsample(D2, size=E1.size()[2:], mode='bilinear')+G1[:, 0,:,:].unsqueeze(1).repeat(1,64,1,1)*T1)
        ################################Gated Parallel&Dual branch residual fuse#######################################
        output_fpn = F.upsample(D1, size=input.size()[2:], mode='bilinear')
        output_res = self.out_res(torch.cat((D1,F.upsample(G5[:, 1,:,:].unsqueeze(1).repeat(1,64,1,1)*T5,size=E1.size()[2:], mode='bilinear'),F.upsample(G4[:, 1,:,:].unsqueeze(1).repeat(1,64,1,1)*T4,size=E1.size()[2:], mode='bilinear'),F.upsample(G3[:, 1,:,:].unsqueeze(1).repeat(1,64,1,1)*T3,size=E1.size()[2:], mode='bilinear'),F.upsample(G2[:, 1,:,:].unsqueeze(1).repeat(1,64,1,1)*T2,size=E1.size()[2:], mode='bilinear'),F.upsample(G1[:, 1,:,:].unsqueeze(1).repeat(1,64,1,1)*T1,size=E1.size()[2:], mode='bilinear')),1))
        output_res =  F.upsample(output_res,size=input.size()[2:], mode='bilinear')
        pre_sal = output_fpn + output_res
        #######################################################################
        if self.training:
            return output_fpn, pre_sal
        # return F.sigmoid(pre_sal)
        return pre_sal
        # return G1,G2,G3,G4,G5

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)


if __name__ == "__main__":
    model = GateNet().eval()
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model,inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops,params)