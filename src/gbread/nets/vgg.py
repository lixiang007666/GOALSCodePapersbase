import itertools
from inspect import Parameter

import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4),
        )
        self._initialize_weights()

    def forward(self, x):
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        feat1 = self.features[  :4 ](x)
        feat2 = self.features[4 :9 ](feat1)
        feat3 = self.features[9 :16](feat2)
        feat4 = self.features[16:23](feat3)
        feat5 = self.features[23:-1](feat4)
        return [feat1, feat2, feat3, feat4, feat5]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels = 3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
# 512,512,3 -> 512,512,64 -> 256,256,64 -> 256,256,128 -> 128,128,128 -> 128,128,256 -> 64,64,256
# 64,64,512 -> 32,32,512 -> 32,32,512
cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

# def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
#                           missing_keys, unexpected_keys, error_msgs):
#     for hook in self._load_state_dict_pre_hooks.values():
#         hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
#
#     local_name_params = itertools.chain(self._parameters.items(), self._buffers.items())
#     local_state = {k: v.data for k, v in local_name_params if v is not None}
#
#     for name, param in local_state.items():
#         key = prefix + name
#         if key in state_dict:
#             input_param = state_dict[key]
#
#             # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
#             if len(param.shape) == 0 and len(input_param.shape) == 1:
#                 input_param = input_param[0]
#
#             if input_param.shape != param.shape:
#                 # local shape should match the one in checkpoint
#                 error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
#                                   'the shape in current model is {}.'
#                                   .format(key, input_param.shape, param.shape))
#                 continue
#
#             if isinstance(input_param, Parameter):
#                 # backwards compatibility for serialized parameters
#                 input_param = input_param.data
#             try:
#                 param.copy_(input_param)
#             except Exception:
#                 error_msgs.append('While copying the parameter named "{}", '
#                                   'whose dimensions in the model are {} and '
#                                   'whose dimensions in the checkpoint are {}.'
#                                   .format(key, param.size(), input_param.size()))
#         elif strict:
#             missing_keys.append(key)
#
#     if strict:
#         for key, input_param in state_dict.items():
#             if key.startswith(prefix):
#                 input_name = key[len(prefix):]
#                 input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
#                 if input_name not in self._modules and input_name not in local_state:
#                     unexpected_keys.append(key)
def VGG16(pretrained, in_channels = 3, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm = False, in_channels = in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth",
                                              model_dir="./model_data")
        model.load_state_dict(state_dict)
        #model.load_state_dict(torch.load(r"model_data\best_epoch_weights.pth"), strict=True)

    
    del model.avgpool
    del model.classifier
    return model
