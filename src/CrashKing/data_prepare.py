'''
Description :
        0. 通过滑窗方式切割数据；
        1. 滑窗同时保存窗口在原图中坐标，推理时将预测结果重新拼回PNG大图；
'''

import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image
from tqdm import tqdm
import os
from argparse import ArgumentParser
import shutil


if __name__ == "__main__":
    parser = ArgumentParser(description="")
    parser.add_argument("--image_path", type=str, default='../../Train/Image/')
    parser.add_argument("--label_path", type=str, default='../../Train/Layer_Masks/')
    parser.add_argument("--save_dir", type=str, default="../preprocess_train_data/")
    arg = parser.parse_args()

    if os.path.isdir(arg.save_dir):
        shutil.rmtree(arg.save_dir)
    os.makedirs(arg.save_dir)

    image_path = arg.image_path
    save_image_dir = os.path.join(arg.save_dir, "image")
    os.makedirs(save_image_dir)

    if arg.label_path == 'None':
        label_path = None
    else:
        label_path = arg.label_path
        save_label_dir = os.path.join(arg.save_dir, "label")
        os.makedirs(save_label_dir)

    stride = (128, 128)
    target_size = (512, 512)

    csv_file = os.path.join(arg.save_dir, "train.csv")
    if os.path.exists(csv_file):
        os.remove(csv_file)

    for filename in os.listdir(image_path):
        basename, filetype = os.path.splitext(filename)

        image = np.asarray(Image.open(os.path.join(image_path, filename)))
        if label_path is not None:
            label = np.asarray(Image.open(os.path.join(label_path, filename)))

        cnt = 0
        csv_pos_list = []

        # 填充外边界
        target_w, target_h = target_size
        h, w = image.shape[0], image.shape[1]
        new_w = (w // target_w) * target_w if (w // target_w == 0) else (w // target_w + 1) * target_w
        new_h = (h // target_h) * target_h if (h // target_h == 0) else (h // target_h + 1) * target_h
        image = cv.copyMakeBorder(image, 48, 48, 26, 26, cv.BORDER_CONSTANT, 0)
        if label_path is not None:
            label = cv.copyMakeBorder(label, 48, 48, 26, 26, cv.BORDER_CONSTANT, value=(255, 255, 255))

        w_stride, h_stride = stride

        def save(cnt, crop_image, crop_label):
            image_name = os.path.join(save_image_dir, basename + "_" + str(cnt) + ".png")
            cv.imwrite(image_name, crop_image)
            if crop_label is not None:
                label_name = os.path.join(save_label_dir, basename + "_" + str(cnt) + ".png")
                cv.imwrite(label_name, crop_label)


        h, w = image.shape[0], image.shape[1]

        for i in tqdm(range((w - target_w) // w_stride + 1)):
            for j in range((h - target_h) // h_stride + 1):
                topleft_x = i * w_stride
                topleft_y = j * h_stride
                crop_image = image[topleft_y:topleft_y + target_h, topleft_x:topleft_x + target_w]
                crop_label = label[topleft_y:topleft_y + target_h, topleft_x:topleft_x + target_w] \
                    if label_path is not None else None

                if label_path is not None:
                    num_255 = np.sum(crop_label[:, :, 1] == 255)
                    total = np.size(crop_label[:, :, 1])

                    if num_255 / total > 0.99:
                        # 剔除背景占比较大的无效图片
                        continue

                if crop_image.shape[:2] != (target_h, target_h):
                    print(topleft_x, topleft_y, crop_image.shape)

                if np.sum(crop_image) == 0:
                    pass
                else:
                    save(cnt, crop_image, crop_label)
                    csv_pos_list.append([basename + "_" + str(cnt) + ".png", topleft_x, topleft_y, topleft_x + target_w,
                                         topleft_y + target_h])
                    cnt += 1

        csv_pos_list = pd.DataFrame(csv_pos_list)
        if label_path is not None:
            csv_pos_list.to_csv(csv_file, header=False, index=False, mode='a+')
        else:
            csv_file = os.path.join(arg.save_dir, basename + '.csv')
            csv_pos_list.to_csv(csv_file, header=False, index=False)
