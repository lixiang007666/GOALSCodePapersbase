from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        """不直接拼接的原因是下采样4次，缩小了2的4次方倍，如果图片的高和宽不是16的整数倍，
        下采样就有可能出现向下取整的情况，从而导致拼接的大小不一样。
        """
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


# 构建空间位置信息
class Up_site(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up_site, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        self.LSTM = nn.LSTM(in_channels, out_channels, 1, bidirectional=True, batch_first=True)
        self.conv_2 = DoubleConv(in_channels*2, out_channels*2)
    def site_relation(self, x, relation=2):
        for i in range(x.size()[relation]):
            if relation == 2:
                x_ = x[:, :, i, :]
                # 交换两个维度的位置
                x_ = torch.transpose(x_, 1, 2)
                x_, _ = self.LSTM(x_)
                x_ = torch.transpose(x_, 1, 2)
                x[:, :, i, :] = x_
            else:
                x_ = x[:, :, :, i]
                # 交换两个维度的位置
                x_ = torch.transpose(x_, 1, 2)
                x_, _ = self.LSTM(x_)
                x_ = torch.transpose(x_, 1, 2)
                x[:, :, :, i] = x_
        return x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        """不直接拼接的原因是下采样4次，缩小了2的4次方倍，如果图片的高和宽不是16的整数倍，
        下采样就有可能出现向下取整的情况，从而导致拼接的大小不一样。
        """
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # shape = [batch_size, 128, crop_size, crop_size]
        x = torch.cat([x2, x1], dim=1)

        # 构建双向LSTM空间位置关系+++++++++++++++++++++++++++++++++++++++++++++++
        x_weigth = self.site_relation(x, relation=2)
        x_heigth = self.site_relation(x, relation=3)
        x_site = torch.cat([x_weigth, x_heigth], dim=1)
        x_site = self.conv_2(x_site)
        x = torch.cat([x, x_site], dim=1)
        x = self.conv_2(x)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )



class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        """ 前三个down模块通道数后面都会翻倍，但最后一层的通道数是没有翻倍的
            由于双线性插值不会改变chanel数，则用双线性插值需要将通道数除以二
        """
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)   # 不使用位置关系
        # self.up4 = Up_site(base_c * 2, base_c, bilinear)    # 使用位置关系++++++++++++++++++++++++++++++++++++++++++++++++
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}
