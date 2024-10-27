import os
from PIL import Image
import numpy as np
from skimage import morphology



def fillHole(pre_mask, classes, size_b, size_v):
    # pre_mask是包含有黑色孔洞的二值图像
    # 这里将原图像黑白反转
    pre_mask_rever = pre_mask <= classes
    # 这里的min_size是表示需要删除的孔洞的大小，可以根据需要设置
    pre_mask_rever = morphology.remove_small_objects(pre_mask_rever, min_size=size_v)
    # 将删除了小连通域的反转图像再盖回原来的图像中
    pre_mask[pre_mask_rever == False] = 255
    pre_mask[pre_mask_rever == True] = classes

    pre_mask_rever = pre_mask > classes
    pre_mask_rever = morphology.remove_small_objects(pre_mask_rever, min_size=size_b)
    pre_mask[pre_mask_rever == False] = classes
    pre_mask[pre_mask_rever == True] = 255

    return pre_mask

def fill(img, fill_num):
    for i in range(len(img[0])):
        start = False
        end = False
        place = []
        for j in range(len(img)):
            if img[j, i] == 0:
                start = True
            if img[j, i] == 255 and start == True:
                place.append([j, i])
            if img[j, i] == 0 and len(place) != 0:
                place = []
                start = False
            if img[j, i] == 80:
                end = True
                break
        if end == True:
            for k in range(len(place)):
                img[place[k][0], place[k][1]] = fill_num
    return img


def main():
    # 选择测试集和训练集
    rootdir = "./predict_test/Layer_Segmentations"
    all_img = os.listdir("predict_test/Layer_Segmentations_0")

    # rootdir = "./predict_train/Layer_Segmentations"
    # all_img = os.listdir("predict_train/Layer_Segmentations_0")

    for i in range(len(all_img)):
        img0_path = rootdir + "_0/" + all_img[i]
        img80_path = rootdir + "_80/" + all_img[i]
        img160_path = rootdir + "_160/" + all_img[i]

        assert os.path.exists(img0_path), f"Layer_Segmentations_0_epoch=200 image {img0_path} not found."
        assert os.path.exists(img80_path), f"Layer_Segmentations_80 image {img80_path} not found."
        assert os.path.exists(img160_path), f"Layer_Segmentations_160 image {img160_path} not found."
        if "195.png" in str(img0_path):
            img0 = Image.open(img0_path).convert('L')
            img80 = Image.open(img80_path).convert('L')
            img160 = Image.open(img160_path).convert('L')

            img0 = np.array(img0)
            img80 = np.array(img80)
            img160 = np.array(img160)

            # 孔洞填充，最后一个值为孔洞的大小，每个类不一样
            # img0 = fillHole(img0, 0, 500, 500)
            # img80 = fillHole(img80, 80, 200, 200)
            # img160 = fillHole(img160, 160, 4000, 4000)
            # new-未测试出效果
            img0 = fillHole(img0, 0, 800, 800)
            img80 = fillHole(img80, 80, 200, 100)
            # img160 = fillHole(img160, 160, 4000, 600)
            img160 = fillHole(img160, 160, 5000, 3000)

            img0[np.logical_and(img80 == 80, img0 == 255)] = 80
            img0[np.logical_and(img160 == 160, img0 == 255)] = 160

            # 填充中间的缝隙,填充数值用0或者80
            img0 = fill(img0, fill_num=80)

            mask = Image.fromarray(img0)
            print(img0_path[-8:], "完成")
            # 训练集和测试集
            # mask.save(os.path.join("./predict_train/Layer_Segmentations/", img0_path[-8:]))
            mask.save(os.path.join("./predict_test/Layer_Segmentations/", img0_path[-8:]))

    print("Image Composite Accomplish")


if __name__ == '__main__':
    main()