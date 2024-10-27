
import random

from PIL import Image
import os

# dir_root_image = "./goal/training/Fundus_color_images_0"
# dir_root_image_seg = "goal/training/Layer_Masks_0"
#
# files = os.listdir(dir_root_image)
#
# write_file = './goal/training/Fundus_color_images_0/'
# write_file_seg = "./goal/training/Layer_Masks_0/"
#
# for index, file_name in enumerate(files):
#     # print(index)
#     img = Image.open(os.path.join(dir_root_image, file_name))
#
#     seg = Image.open(os.path.join(dir_root_image_seg, file_name))
#
#     imgwidth = img.width
#     imgheight = img.height
#     print(imgwidth,imgheight)
#
#
#     leftwidth1 = 0
#     upperwidth1 = 0
#
#     rightwidth1 = imgwidth
#     underwidth1 = imgheight - 250
#
#     cropped1 = img.crop((leftwidth1, upperwidth1, rightwidth1, underwidth1))
#     cropped1_seg = seg.crop((leftwidth1, upperwidth1, rightwidth1, underwidth1))
#
#     print(file_name)
#
#
#     cropped1.save(write_file + file_name)
#     cropped1_seg.save(write_file_seg + file_name)
#
#
#
#
#     leftwidth1 = 0
#     upperwidth1 = 50
#
#     rightwidth1 = imgwidth
#     underwidth1 = imgheight - 200
#
#     cropped2 = img.crop((leftwidth1, upperwidth1, rightwidth1, underwidth1))
#     cropped2_seg = seg.crop((leftwidth1, upperwidth1, rightwidth1, underwidth1))
#
#     file_name1 = str(int(file_name.split('.')[0]) +100).zfill(4) + '.png'
#
#     print(file_name1)
#
#
#
#     cropped2.save(write_file + file_name1)
#     cropped2_seg.save(write_file_seg + file_name1)
#
#
#
#
#     leftwidth1 = 0
#     upperwidth1 = 100
#
#     rightwidth1 = imgwidth
#     underwidth1 = imgheight - 150
#
#     cropped3 = img.crop((leftwidth1, upperwidth1, rightwidth1, underwidth1))
#     cropped3_seg = seg.crop((leftwidth1, upperwidth1, rightwidth1, underwidth1))
#
#     file_name2 = str(int(file_name.split('.')[0]) + 200).zfill(4) + '.png'
#     print(file_name2)
#
#
#     cropped2.save(write_file + file_name2)
#     cropped2_seg.save(write_file_seg + file_name2)

#
#
# dir_root_image = "./goal/training/Fundus_color_images_0"
# dir_root_image_seg = "goal/training/Layer_Masks_0"
#
# files = os.listdir(dir_root_image)
#
# write_file = './goal/training/Fundus_color_images_0/'
# write_file_seg = "./goal/training/Layer_Masks_0/"
#
#
# for index, file_name in enumerate(files):
#     # print(index)
#     img = Image.open(os.path.join(dir_root_image, file_name))
#
#     seg = Image.open(os.path.join(dir_root_image_seg, file_name))
#
#     imgwidth = img.width
#     imgheight = img.height
#     # print(imgwidth,imgheight)
#
#     cropped3 = img.transpose(Image.FLIP_LEFT_RIGHT)
#     cropped3_seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
#
#     file_name3 = str(int(file_name.split('.')[0]) + 300).zfill(4) + '.png'
#     print(file_name3)
#
#
#     cropped3.save(write_file + file_name3)
#     cropped3_seg.save(write_file_seg + file_name3)
#




dir_root_image = "./goal/testing/Fundus_color_images_0"

files = os.listdir(dir_root_image)

write_file = './goal/testing/Fundus_color_images/'


for index, file_name in enumerate(files):
    # print(index)
    img = Image.open(os.path.join(dir_root_image, file_name))

    imgwidth = img.width
    imgheight = img.height
    print(imgwidth,imgheight)

    leftwidth1 = 0
    upperwidth1 = 50

    rightwidth1 = imgwidth
    underwidth1 = imgheight - 200

    cropped4 = img.crop((leftwidth1, upperwidth1, rightwidth1, underwidth1))
    cropped4.save(write_file + file_name)


