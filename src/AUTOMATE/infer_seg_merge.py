import os

import cv2
import numpy as np
from skimage import measure, morphology


def keep_max_area(mask: np.ndarray):
    label = measure.label(mask)
    regions = measure.regionprops(label)
    max_idx = np.argmax([region.area for region in regions])
    label[label != max_idx + 1] = 0
    return np.uint8(label)


def fill_between_cls12_area_to_cls1(mask: np.ndarray):
    mask_t = morphology.remove_small_holes(mask != 255, area_threshold=860).astype(np.uint8)
    mask[np.logical_and(mask == 255, mask_t != 0)] = 0
    return mask


mask_name_list = [path for path in os.listdir('GOALS2022-Validation/GOALS2022-Validation/Image') if '.png' in path]

os.makedirs('results/Layer_Segmentations', exist_ok=True)

for mask_name in mask_name_list:
    cls3 = cv2.imread(f'data/cls3/pseudo_color_prediction/{mask_name}', cv2.IMREAD_GRAYSCALE)
    cls2 = cv2.imread(f'data/cls2/pseudo_color_prediction/{mask_name}', cv2.IMREAD_GRAYSCALE)
    cls1 = cv2.imread(f'data/cls123/pseudo_color_prediction/{mask_name}', cv2.IMREAD_GRAYSCALE)

    cls3 = keep_max_area(np.uint8(cls3 == 160))
    cls2 = keep_max_area(np.uint8(cls2 == 80))
    cls1 = keep_max_area(np.uint8(cls1 == 0))

    mask = np.ones_like(cls1, dtype='uint8') * 255
    mask[cls1 != 0] = 0
    mask[cls2 != 0] = 80
    mask = fill_between_cls12_area_to_cls1(mask)
    mask[cls3 != 0] = 160

    cv2.imwrite(os.path.join('results/Layer_Segmentations', mask_name), mask)
