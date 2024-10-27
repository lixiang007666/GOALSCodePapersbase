import os
import threading
import cv2
import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects
from argparse import ArgumentParser
from PIL import Image
import datetime


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')

    # 对官方提供的原始label做了调整，便于二值图像后处理
    y[y == 0] = 1
    y[y == 80] = 2
    y[y == 160] = 3
    y[y == 255] = 0

    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class MyThread(threading.Thread):

    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def label_vis(label, img=None, alpha=0.5):
    '''
    :param label:原始标签
    :param img: 原始图像
    :param alpha: 透明度
    :return: 可视化标签
    '''
    r = np.where(label == 0, 255, 0)
    g = np.where(label == 80, 255, 0)
    b = np.where(label == 160, 255, 0)

    anno_vis = np.dstack((b, g, r)).astype(np.uint8)

    anno_vis[:, :, 0] = anno_vis[:, :, 0]
    anno_vis[:, :, 1] = anno_vis[:, :, 1]
    anno_vis[:, :, 2] = anno_vis[:, :, 2]
    if img is None:
        return anno_vis
    else:
        overlapping = cv2.addWeighted(img, alpha, anno_vis, 1 - alpha, 0)
        return overlapping


def remove_small_objects_and_holes(class_type, label, min_size, area_threshold, in_place=True):
    if class_type != 2:
        label = remove_small_objects(label == 1, min_size=min_size, connectivity=1, in_place=in_place)
        label = remove_small_holes(label == 1, area_threshold=area_threshold, connectivity=1, in_place=in_place)

    return label


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='UNet', help='Name of model')
    parser.add_argument("--source_image_path", type=str, default='../Validation/Image/')
    parser.add_argument("--predicted_image_path", type=str, default='./predict_results/2022y07m01d_22h59m25s_unet/Layer_Segmentations')
    parser.add_argument("--threshold", type=int, default=2000)
    parser.add_argument("--save_dir", type=str, default='./postprocess_results')

    args = parser.parse_args()

    source_image_path = args.source_image_path
    predicted_mask_path = args.predicted_image_path
    threshold = args.threshold

    # save_dir
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Yy%mm%dd_%Hh%Mm%Ss_')
    model_tag = args.model.lower()

    layer_segmentationssave_dir = os.path.join(args.save_dir, time_str+model_tag, 'Layer_Segmentations')
    os.makedirs(layer_segmentationssave_dir, exist_ok=True)
    show_layer_segmentationssave_dir = os.path.join(args.save_dir, time_str+model_tag, 'Show_Layer_Segmentations')
    os.makedirs(show_layer_segmentationssave_dir, exist_ok=True)

    for filename in os.listdir(predicted_mask_path):
        if filename[-4:] == '.png':

            image_path = os.path.join(source_image_path, filename)
            source_image = cv2.imread(image_path)

            mask_path = os.path.join(predicted_mask_path, filename)
            mask = np.asarray(Image.open(mask_path))

            label = to_categorical(mask, num_classes=4, dtype='uint8')

            threading_list = []
            for i in range(4):
                t = MyThread(remove_small_objects_and_holes, args=(i, label[:, :, i], threshold, threshold, True))
                threading_list.append(t)
                t.start()

            # 等待所有线程运行完毕
            result = []
            for t in threading_list:
                t.join()
                result.append(t.get_result()[:, :, None])

            label = np.concatenate(result, axis=2)

            label = np.argmax(label, axis=2).astype(np.uint8)

            label[label == 0] = 255
            label[label == 1] = 0
            label[label == 2] = 80
            label[label == 3] = 160

            cv2.imwrite(os.path.join(layer_segmentationssave_dir, filename[:-4] + ".png"), label)

            mask_vis = label_vis(label, source_image)
            cv2.imwrite(os.path.join(show_layer_segmentationssave_dir, filename[:-4] + ".png"), mask_vis)
            print("{} saved".format(filename[:-4] + ".png"))
