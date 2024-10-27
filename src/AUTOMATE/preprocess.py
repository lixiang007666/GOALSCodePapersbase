import glob
import os
from tqdm import tqdm
import cv2


def gray_images(image_dir: str):
    image_path_list = glob.glob(os.path.join(image_dir, '*.png'))
    for image_path in tqdm(image_path_list, desc=image_dir):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(image_path, image)


def relabel_labels(label_dir: str):
    label_path_list = glob.glob(os.path.join(label_dir, '*.png'))
    for label_path in tqdm(label_path_list, desc=label_dir):
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label[label == 0] = 1
        label[label == 80] = 2
        label[label == 160] = 3
        label[label == 255] = 0
        cv2.imwrite(label_path, label)


if __name__ == '__main__':
    gray_images('GOALS2022-Train/Train/Image')
    relabel_labels('GOALS2022-Train/Train/Layer_Masks')

    gray_images('GOALS2022-Validation/GOALS2022-Validation/Image')
