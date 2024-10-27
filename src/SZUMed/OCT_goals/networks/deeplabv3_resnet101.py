
import torch

from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, \
   deeplabv3_resnet50, deeplabv3_resnet101

# from torchvision.models.segmentation import
#
# model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)  # pretrained=True 设为预训练模式
# model.eval()


def deeplabv3_resnet10one(num_classes=2, include_top=True):

    net = deeplabv3_resnet101(num_classes=num_classes)
    # net_weights = net.state_dict()
    # model_weights_path = "../pretrained_ckpt/deeplabv3_resnet101_coco-586e9e4e.pth"
    # pre_weights = torch.load(model_weights_path)
    # # delete classifier weights
    # # 这种方法主要是遍历字典，.pth文件（权重文件）的本质就是字典的存储
    # # 通过改变我们载入的权重的键值对，可以和当前的网络进行配对的
    # # 这里举到的例子是对"classifier"结构层的键值对剔除，或者说是不载入该模块的训练权重，这里的"classifier"结构层就是最后一部分分类层
    # pre_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}
    # # 如果修改了载入权重或载入权重的结构和当前模型的结构不完全相同，需要加strict=False，保证能够权重载入
    # net.load_state_dict(pre_dict, strict=False)

    return net

def deeplabv3_resnet5zero(num_classes=2, include_top=True):

    net = deeplabv3_resnet50(num_classes=num_classes)

    # model_weights_path = "../pretrained_ckpt/deeplabv3_resnet101_coco-586e9e4e.pth"
    # pre_weights = torch.load(model_weights_path)
    # # delete classifier weights
    # # 这种方法主要是遍历字典，.pth文件（权重文件）的本质就是字典的存储
    # # 通过改变我们载入的权重的键值对，可以和当前的网络进行配对的
    # # 这里举到的例子是对"classifier"结构层的键值对剔除，或者说是不载入该模块的训练权重，这里的"classifier"结构层就是最后一部分分类层
    # pre_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}
    # # 如果修改了载入权重或载入权重的结构和当前模型的结构不完全相同，需要加strict=False，保证能够权重载入
    # net.load_state_dict(pre_dict, strict=False)

    return net



def fcn_resnet5zero(num_classes=2, include_top=True):

    net = fcn_resnet50(num_classes=num_classes)

    return net


if __name__ == "__main__":

    net = deeplabv3_resnet50(num_classes=4)
    # net.load_from(weights=np.load(config_vit.pretrained_path))
    input = torch.rand(16,3, 448, 896)

    logits = net(input)

    print(logits['out'].shape)








