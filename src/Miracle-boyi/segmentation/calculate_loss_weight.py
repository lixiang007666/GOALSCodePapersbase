import os
from PIL import Image
import numpy as np

def main():
    # 选择测试集和训练集
    rootdir = "./data/Train/Layer_Masks/"
    all_img = os.listdir("data/Train/Layer_Masks")

    # rootdir = "./predict_train/Layer_Segmentations"
    # all_img = os.listdir("predict_train/Layer_Segmentations_0")

    number1 = 0
    number0 = 0

    for i in range(len(all_img)):
        img_path = rootdir + all_img[i]

        assert os.path.exists(img_path), f"Layer_Segmentations_0_epoch=200 image {img_path} not found."

        img = Image.open(img_path).convert('L')
        img = np.array(img)
        img[img == 0] = 1
        img[img == 80] = 0
        img[img == 160] = 0
        img[img == 255] = 0

        num1 = np.sum(img == 1)
        num0 = np.sum(img == 0)

        if i == 0:
            number1 = num1
            number0 = num0
        else:
            number1 = (number1 + num1)/2
            number0 = (number0 + num0)/2

    print(number1/(number0+number1))
    print(number0/(number0+number1))

if __name__ == '__main__':
    main()