import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from src import UNet
from src.unet_plus import NestedUNet
import matplotlib.pyplot as plt

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 1  # exclude background
    weights_path = "weights_3090/best_model_0_epoch=400_3090.pth"
    # 对比测试集的标准情况
    all_img = os.listdir("./data/GOALS2022-Validation/Image")
    rootdir = "./data/GOALS2022-Validation/Image/"

    # 对比训练集的标准情况
    # all_img = os.listdir("./data/Train/Image")
    # rootdir = "./data/Train/Image/"

    for i in range(len(all_img)):

        # img_path = "test/0001.png"
        img_path = rootdir + all_img[i]

        roi_mask_path = "./DRIVE/test/mask/01_test_mask.gif"
        assert os.path.exists(weights_path), f"weights {weights_path} not found."
        assert os.path.exists(img_path), f"image {img_path} not found."
        assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

        # mean = (0.709, 0.381, 0.224)
        # std = (0.127, 0.079, 0.043)

        mean = (0.226, 0.226, 0.226)
        std = (0.160, 0.160, 0.160)

        # mean = (0.485, 0.456, 0.406)
        # std = (0.229, 0.224, 0.225)

        # get devices
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using {} device.".format(device))

        # create model
        # unet模型
        model = UNet(in_channels=3, num_classes=classes+1, base_c=64)
        # unet++模型
        # model = NestedUNet(in_channels=3, num_classes=classes+1)

        # load weights
        model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
        model.to(device)

        # load roi mask
        roi_img = Image.open(roi_mask_path).convert('L')
        roi_img = np.array(roi_img)

        # load image
        original_img = Image.open(img_path).convert('RGB')

        # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))
            # unet模型的预测
            prediction = output['out'].argmax(1).squeeze(0)

            # unet++模型的预测
            # prediction = ((output['out'] + output['out3'] + output['out2'] + output['out1'])/4).argmax(1).squeeze(0)

            prediction = prediction.to("cpu").numpy().astype(np.uint8)


            # 将前景对应的像素值改成255(白色)
            # 预测第一个类-----0
            # prediction[prediction == 0] = 255
            # prediction[prediction == 1] = 0

            # 预测第二个类-----80
            prediction[prediction == 0] = 255
            prediction[prediction == 1] = 80

            # 预测第三个类-----160
            # prediction[prediction == 0] = 255
            # prediction[prediction == 1] = 160


            # 将不敢兴趣的区域像素设置成0(黑色)
            # prediction[roi_img == 0] = 0
            mask = Image.fromarray(prediction)

            # mask.save("result")

            # 训练集和测试集文件夹的分开预测
            # root_path = "./predict_train"
            root_path = "./predict_test"

            # 预测第一个类-----0
            # mask.save(os.path.join(root_path + "/Layer_Segmentations_0/", img_path[-8:]))

            # 预测第二个类-----80
            mask.save(os.path.join(root_path + "/Layer_Segmentations_80/", img_path[-8:]))

            # 预测第三个类-----160
            # mask.save(os.path.join(root_path + "/Layer_Segmentations_160/", img_path[-8:]))




if __name__ == '__main__':
    main()
