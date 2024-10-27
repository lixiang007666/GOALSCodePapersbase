import torch
from torch import nn
from ..blocks.weighted_block import WeightedBlock
from torchvision.models.resnet import Bottleneck


class WeightedAggregationModule(nn.Module):
    """Weighted Aggregation Module
    """
    def __init__(self):
        super().__init__()
        out_channels = 4

        self.output_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, out_channels, kernel_size=1, stride=1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

    def forward(self, x_3, x_e):

        weight_c = torch.cat((x_3, x_e), dim=1)
        return self.output_conv(weight_c)
    #
    #     in_1_channels = 128
    #     in_2_channels = 64
    #     in_3_channels = 16
    #     in_e_channels = 64
    #     out_1_channels = 64
    #     out_channels = 4
    #
    #
    #     self.weight_1 = nn.Sequential(
    #         WeightedBlock(in_1_channels, out_1_channels),
    #         nn.Upsample(scale_factor=2, mode='bilinear'),
    #     )
    #
    #     self.weight_2 = WeightedBlock(in_2_channels, out_1_channels)
    #     self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear')
    #
    #     self.weight_3 = WeightedBlock(in_3_channels, out_1_channels)
    #     self.output_conv = nn.Sequential(
    #         nn.Conv2d(out_1_channels + in_e_channels, out_channels, kernel_size=1),  # 1x1
    #         nn.BatchNorm2d(out_channels),
    #         # nn.ReLU(inplace=True),
    #         # nn.Softmax(dim=1)
    #     )
    #
    # def forward(self, x_1, x_2, x_3, x_e):
    #     weight_1 = self.weight_1(x_1)
    #     weight_2 = self.up_2(weight_1 + self.weight_2(x_2))
    #     weight_3 = weight_2 + self.weight_3(x_3)
    #     # print(weight_3.shape,x_e.shape)
    #     weight_c = torch.cat((weight_3, x_e), dim=1)
    #     return self.output_conv(weight_c)




