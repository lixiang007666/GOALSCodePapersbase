

from .Efficientunet.efficientunet import *

import torch

def Efficientunet_b4(num_classes=2):
    # print(111)
    net = get_efficientunet_b4(out_channels=num_classes,pretrained=False)

    return net
