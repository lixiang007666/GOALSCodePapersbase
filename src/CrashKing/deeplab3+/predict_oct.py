import argparse

import torch
from torch.utils.data import DataLoader

import datetime
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

from utils.data_loading import Inference_Dataset
from models import UNet

torch.backends.cudnn.benchmark = True


def create_zeros_png(image_w, image_h):
    '''Description:
        创造一个空白图像，将滑窗预测结果逐步填充至空白图像中；
    '''
    new_h, new_w = 896, 1152
    zeros = (new_h, new_w)
    zeros = np.ones(zeros, np.uint8) * 255
    return zeros


def tta_forward(dataloader, model, png_shape, device=None):
    image_w, image_h = png_shape
    predict_png = create_zeros_png(image_w, image_h)
    model = model.eval()

    with torch.no_grad():
        for (image, pos_list) in tqdm(dataloader):
            # forward --> predict
            image = image.to(device)

            predict_1 = model(image)

            predict_2 = model(torch.flip(image, [-1]))
            predict_2 = torch.flip(predict_2, [-1])

            predict_3 = model(torch.flip(image, [-2]))
            predict_3 = torch.flip(predict_3, [-2])

            predict_4 = model(torch.flip(image, [-1, -2]))
            predict_4 = torch.flip(predict_4, [-1, -2])

            predict_list = predict_1 + predict_2 + predict_3 + predict_4
            predict_list = torch.argmax(predict_list.cpu(), 1).byte().numpy()  # n x h x w

            batch_size = predict_list.shape[0]
            for i in range(batch_size):

                predict = predict_list[i]
                predict[predict == 1] = 80
                predict[predict == 2] = 160
                predict[predict == 3] = 255

                pos = pos_list[i, :]
                [topleft_x, topleft_y, buttomright_x, buttomright_y] = pos

                if (buttomright_x - topleft_x) == 512 and (buttomright_y - topleft_y) == 512:
                    # 每次预测只保留图像中心区域预测结果
                    predict_png[topleft_y + 48:buttomright_y - 48, topleft_x + 26:buttomright_x - 26] = predict[
                                                                                                        48:512 - 48,
                                                                                                        26:512 - 26
                                                                                                        ]
                else:
                    raise ValueError(
                        "target_size!=512， Got {},{}".format(buttomright_x - topleft_x, buttomright_y - topleft_y))

    h, w = predict_png.shape
    predict_png = predict_png[48:h - 48, 26:w - 26]  # 去除整体外边界

    return predict_png


def get_args():
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('--model', '-m', metavar='M', type=str, default='UNet', help='Name of model')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/checkpoint_epoch50.pth')
    parser.add_argument('--root_path', type=str, default='./preprocess_val_data')
    parser.add_argument('--save_dir', type=str, default='./predict_results')
    parser.add_argument('--batch_size', '-b', metavar='B', type=int, default=1, help='Batch size')

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    checkpoint_path = args.checkpoint_path

    # save_dir
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Yy%mm%dd_%Hh%Mm%Ss_')
    model_tag = args.model.lower()
    save_dir = os.path.join(args.save_dir, time_str+model_tag, 'Layer_Segmentations')
    os.makedirs(save_dir, exist_ok=True)

    model = UNet(in_channels=3, num_classes=4)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    root_path = args.root_path
    image_dir = os.path.join(root_path, 'image')  # 测试图像路径
    for filename in os.listdir(root_path):
        if filename[-4:] == '.csv':
            csv_file = os.path.join(root_path, filename)

            test_set = Inference_Dataset(image_dir, csv_file)
            loader_args = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_set, shuffle=True, **loader_args)
            predict_png = tta_forward(test_loader, model, device=device, png_shape=(800, 1100))

            pil_image = Image.fromarray(predict_png)
            pil_image.save(os.path.join(save_dir, filename[:-4] + ".png"))
            print("{} saved".format(filename[:-4] + ".png"))
