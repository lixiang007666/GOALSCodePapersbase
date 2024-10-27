import torch
from torch import nn


from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding


# from args import ARGS
from torchvision.models.resnet import Bottleneck


class EAM_Module(Module):
    """ Edge attention module in Supervised Edge Attention Network for Accurate Image Instance Segmentation"""

    def __init__(self, in_dim):
        super(EAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        out = self.conv(out)
        return out


class EdgeGuidanceModule(nn.Module):
    """Edge Guidance Module
    """
    def __init__(self):
        super().__init__()



        self.input_conv_1 = nn.Sequential(
                                nn.Conv2d(256, 256, 1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU()
                          )

        self.dncoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            # nn.Conv2d(256, 2, kernel_size=1, stride=1),
            # nn.UpsamplingBilinear2d(scale_factor=2)

        )
        self.output_edge_conv = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1, stride=1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1),
            # nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.EAM = EAM_Module(in_dim=256)

    def forward(self, x_1):
        # print(x_1.shape)

        x = self.input_conv_1(x_1)
        x = self.EAM(x)
        x = self.dncoder(x)

        return self.output_edge_conv(x), self.output_conv(x)


    
