import os
from PIL import Image
import numpy as np


def main():
    img_channels = 3
    img_dir = "./data/Train/Image"
    roi_dir = "./data/Train/Layer_Masks"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    assert os.path.exists(roi_dir), f"roi dir: '{roi_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".png")]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        ori_path = os.path.join(roi_dir, img_name.replace(".tif", "_mask.gif"))

        # 原版
        img = np.array(Image.open(img_path).convert('RGB')) / 255.
        roi_img = np.array(Image.open(ori_path).convert('L'))
        roi_img[roi_img == 0] = 255
        roi_img[roi_img == 80] = 255
        roi_img[roi_img == 160] = 255

        img = img[roi_img == 255]
        cumulative_mean += img.mean(axis=0)
        cumulative_std += img.std(axis=0)

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == '__main__':
    main()
