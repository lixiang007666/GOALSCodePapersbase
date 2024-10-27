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

img_path = 'predict_test/Layer_Segmentations_160/0195.png'
img = Image.open(img_path).convert('L')
img = np.array(img)
print(img.shape)

# for i in range(img.shape[1]):   # 1000
#     find = False
#     change = False
#     location = []
#     for j in range(img.shape[0]):   # 800
#         if img[j, i]==0 and find==False:
#             location.append([j, i])
#
#         if len(location)!=0 and img[j, i] == 255:
#             find = True
#
#         if find==True and img[j, i]==0:
#             change = True
#
#         if img[j, i]==80 or img[j, i]==160:
#             break
#     if change==True:
#         for k in range(len(location)):
#             img[location[k][0], location[k][1]] = 255

# for i in range(img.shape[1]):   # 1000
#     find = 0
#     change = False
#     location = []
#     for j in range(img.shape[0]):   # 800
#         if img[j, i]==0 and find==0:
#             location.append([j, i])
#         elif img[j, i] == 255 and len(location)!=0:
#             find=1
#         if find==1 and img[j, i]==0:
#             change=True
#             break
#         if img[j, i]==80 or img[j, i]==160:
#             break
#     if change==True:
#         for k in range(len(location)):
#             img[location[k][0], location[k][1]] = 160

# 处理160的洞


# for i in range(img.shape[1]):   # 1000
#     find = 0
#     bai = False
#     change = False
#     location = []
#     for j in range(img.shape[0]):   # 800
#         if img[j, i]==160 and find==0:
#             find = 1
#         if img[j, i]==255 and find==1:
#             location.append([j, i])
#         if len(location)!=0 and img[j, i]==160:
#             break
#     if len(location)<200:
#         for k in range(len(location)):
#             img[location[k][0], location[k][1]] = 160

# for i in range(img.shape[1]):   # 1000
#     find = 0
#     bai = False
#     change = False
#     location = []
#     for j in range(img.shape[0]):   # 800
#         if img[j, i]==160 and find==0:
#             find = 1
#         if img[j, i]==255 and find==1:
#             location.append([j, i])
#         if len(location)!=0 and img[j, i]==160:
#             break
#     if len(location)<200:
#         for k in range(len(location)):
#             img[location[k][0], location[k][1]] = 160
#
# for i in range(img.shape[1]):   # 1000
#     find = 0
#     bai = False
#     change = False
#     location = []
#     for j in range(img.shape[0]):   # 800
#         if img[j, i]==160 and find==0:
#             find = 1
#         if img[j, i]==255 and find==1:
#             location.append([j, i])
#         if len(location)!=0 and img[j, i]==160:
#             break
#     if len(location)<200:
#         for k in range(len(location)):
#             img[location[k][0], location[k][1]] = 160
for i in range(478, 512):
    for j in range(38):   # 800
        if img[i, j]==160 :
            img[i, j]=255


mask = Image.fromarray(img)
mask.show()
mask.save("./predict_test/Layer_Segmentations_160/"+img_path[-8:])