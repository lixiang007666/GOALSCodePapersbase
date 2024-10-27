# task1网络模型
from turtle import forward
import cv2
import torch
import torch.nn as nn
from torchsummary import summary

# unet
class DownsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        # 下采样过程中通道数变化：
        # 3-64-64-64
        # 64-128-128-128
        # 128-256-256-256
        # 256-512-512-512
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_RELU_2 = nn.Sequential( # 增加特征维度
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        self.downsample = nn.Sequential( # 下采样
            # nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU()
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        # out通过残差连接输出到对称的深层，out_2输出到下一层
        out = self.Conv_BN_RELU_2(x)
        out_2 = self.downsample(out)

        return out, out_2

class UpSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        # 上采样过程中通道数变化：
        # 512-1024-1024-512
        # 1024-512-512-256
        # 512-256-256-128
        # 256-128-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2, out_channels=out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, out):
        # x：输入卷积层
        # out：与上采样层进行cat
        x_out = self.Conv_BN_ReLU_2(x)
        x_out = self.upsample(x_out)
        cat_out = torch.cat((x_out, out), dim=1)
        return cat_out

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        out_channels = [64, 128, 256, 512, 1024]
        # 下采样
        self.d1 = DownsampleLayer(3, out_channels[0]) # 3-64
        self.d2 = DownsampleLayer(out_channels[0], out_channels[1]) # 64-128
        self.d3 = DownsampleLayer(out_channels[1], out_channels[2]) # 128-256
        self.d4 = DownsampleLayer(out_channels[2], out_channels[3]) # 256-512
        # 上采样
        self.u1 = UpSampleLayer(out_channels[3], out_channels[3])# 512-1024-1024-512
        self.u2 = UpSampleLayer(out_channels[4], out_channels[2])# 1024-512-512-256
        self.u3 = UpSampleLayer(out_channels[3], out_channels[1])# 512-256-256-128
        self.u4 = UpSampleLayer(out_channels[2], out_channels[0])# 256-128-128-64

        # 输出
        self.o = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels=4, kernel_size=3, stride=1, padding=1), # out_channels=4为输出分割类别
            # nn.Sigmoid()
        )
    def forward(self, x):
        out_residual1, out1 = self.d1(x)
        out_residual2, out2 = self.d2(out1)
        out_residual3, out3 = self.d3(out2)
        out_residual4, out4 = self.d4(out3)
        out5 = self.u1(out4, out_residual4)
        out6 = self.u2(out5, out_residual3)
        out7 = self.u3(out6, out_residual2)
        out8 = self.u4(out7, out_residual1)
        out = self.o(out8)
        return out


if __name__=='__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = UNet().to(device)
    img = torch.rand(1,3,256,256).to(device)
    for i in range(100):
        y = net(img)
    print(y.shape)
    # cv2.imwrite('out.png', y.cpu().detach().numpy())
    summary(net, input_size=(3, 256,256))
