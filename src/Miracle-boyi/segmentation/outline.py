import torch.nn.functional as F
import  torch
from torchvision import transforms as T
def generate_detail_gt(gtmasks):
    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)

    # fuse_kernel = torch.nn.Parameter(
    #     torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
    #     dtype=torch.float32).reshape(1, 3, 1, 1).type(
    #     torch.cuda.FloatTensor)
    # )

    boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0
    return boundary_targets
    # boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), laplacian_kernel, stride=2, padding=1)
    # boundary_targets_x2 = boundary_targets_x2.clamp(min=0)
    #
    # boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), laplacian_kernel, stride=4, padding=1)
    # boundary_targets_x4 = boundary_targets_x4.clamp(min=0)
    #
    # boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), laplacian_kernel, stride=8, padding=1)
    # boundary_targets_x8 = boundary_targets_x8.clamp(min=0)

    # boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
    # boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
    # boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')
    #
    # boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
    # boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0
    #
    # boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
    # boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0
    #
    # boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
    # boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0

    # boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=1)
    #
    # boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
    # boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, fuse_kernel)
    #
    # boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
    # boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0

    # if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
    #         boundary_logits = F.interpolate(boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)

    # return boudary_targets_pyramids


if __name__ == "__main__":

    from PIL import Image
    # import PIL.Image.T
    gtmask = Image.open("./data/Train/Layer_Masks/0001.png").convert('L')
    # gtmask.show()
    gtmask = T.ToTensor()(gtmask)
    print(gtmask.shape)

    # gtmask = torch.rand(1, 128, 128)

    o = generate_detail_gt(gtmask)
    o = o[0]

    print(o.shape)
    show = T.ToPILImage()(o[0])
    show.show()
    # for i in range(3):
    #     show = T.ToPILImage()(o[i])
    #     show.show()
    #
    # print("---")
