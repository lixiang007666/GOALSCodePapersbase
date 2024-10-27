import os
from skimage import morphology
from PIL import Image
import numpy as np
from scipy import ndimage

def fillHole(pre_mask, classes, size):
    # pre_mask是包含有黑色孔洞的二值图像
    # 这里将原图像黑白反转
    pre_mask_rever = pre_mask <= classes
    # 这里的min_size是表示需要删除的孔洞的大小，可以根据需要设置
    pre_mask_rever = morphology.remove_small_objects(pre_mask_rever, min_size=size)
    # 将删除了小连通域的反转图像再盖回原来的图像中
    pre_mask[pre_mask_rever == False] = 255
    pre_mask[pre_mask_rever == True] = classes

    pre_mask_rever = pre_mask > classes
    pre_mask_rever = morphology.remove_small_objects(pre_mask_rever, min_size=size)
    pre_mask[pre_mask_rever == False] = classes
    pre_mask[pre_mask_rever == True] = 255

    return pre_mask


def main():
    # 选择测试集和训练集
    rootdir = "./predict_test/Layer_Segmentations"
    all_img = os.listdir("predict_test/Layer_Segmentations_0")

    for i in range(len(all_img)):
        img0_path = rootdir + "_0/" + all_img[i]
        img80_path = rootdir + "_80/" + all_img[i]
        img160_path = rootdir + "_160/" + all_img[i]

        assert os.path.exists(img0_path), f"Layer_Segmentations_0_epoch=200 image {img0_path} not found."
        assert os.path.exists(img80_path), f"Layer_Segmentations_80 image {img80_path} not found."
        assert os.path.exists(img160_path), f"Layer_Segmentations_160 image {img160_path} not found."

        img0 = Image.open(img0_path).convert('L')
        img80 = Image.open(img80_path).convert('L')
        img160 = Image.open(img160_path).convert('L')

        img0 = np.array(img0)
        img80 = np.array(img80)
        img160 = np.array(img160)

        img0 = fillHole(img0, 0, 300)
        img0 = Image.fromarray(img0)
        img0.save(os.path.join("./remove_test/", img0_path[-8:]))

        # img80 = fillHole(img80, 80, 200)
        # img80 = Image.fromarray(img80)
        # img80.save(os.path.join("./remove_test/", img80_path[-8:]))

        # img160 = fillHole(img160, 160, 500)
        # img160 = Image.fromarray(img160)
        # img160.save(os.path.join("./remove_test/", img160_path[-8:]))


if __name__ == '__main__':
    main()
