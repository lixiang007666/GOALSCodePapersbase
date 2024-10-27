import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101

# 网络模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.feature = resnet50(pretrained=True) # 移除最后一层全连接
        # self.feature = resnet101(pretrained=True, num_classes=2) # 移除最后一层全连接
        # fc_inputs = self.feature.fc.in_features
        self.fc1 = nn.Linear(1000, 1024) # resnet50后面连接了一个fc层，输出维度是1000
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, img):
        feature = self.feature(img)
        out1 = self.fc1(feature)
        logit = self.fc2(out1)

        return logit