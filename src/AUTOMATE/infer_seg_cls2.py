import glob
import os

import cv2
import numpy as np
import paddleseg
from paddleseg import transforms as T
from paddleseg.core import predict
from scipy.stats import mode
from tqdm import tqdm

model = paddleseg.models.U2Net(num_classes=2)

image_path_list = glob.glob('GOALS2022-Validation/GOALS2022-Validation/Image/*.png')

eval_transforms = [
    T.Padding(target_size=(1120, 800)),
    T.Normalize(
        mean=[0.2297999174664977, 0.2297999174664977, 0.2297999174664977],
        std=[0.16316278003205756, 0.16316278003205756, 0.16316278003205756])
]

for fold_num in range(1, 6):
    predict(
        model=model,
        model_path=f'models_seg/cls2_U2Net_Dice_Norm_Fold{fold_num}/best_model/model.pdparams',
        transforms=T.Compose(eval_transforms),
        image_list=image_path_list,
        save_dir=f'data/cls2_KFold_TMP/Fold{fold_num}',
        aug_pred=False,
        flip_horizontal=True,
        flip_vertical=False,
        custom_color=[255, 255, 255, 80, 80, 80])

save_dir = 'data/cls2/pseudo_color_prediction'
os.makedirs(save_dir, exist_ok=True)

image_name_list = os.listdir('GOALS2022-Validation/GOALS2022-Validation/Image')
image_name_list = sorted([name for name in image_name_list if '.png' in name])

for image_name in tqdm(image_name_list):
    label_list = []
    for fold_num in range(1, 6):
        label_fold_path = os.path.join(f'data/cls2_KFold_TMP/Fold{fold_num}/pseudo_color_prediction', image_name)
        label_list.append(cv2.imread(label_fold_path, cv2.IMREAD_GRAYSCALE))
    label_list = np.asarray(label_list)

    mean = np.mean(label_list, axis=0)
    row, col = np.where(
        np.logical_and(
            mean != label_list[0],
            np.logical_and(
                mean != label_list[1],
                np.logical_and(
                    mean != label_list[2],
                    np.logical_and(
                        mean != label_list[3],
                        mean != label_list[4])))))
    label = mean.astype(np.uint8)
    for r, c in zip(row, col):
        label[r, c] = int(mode(label_list[:, r, c])[0][0])
    cv2.imwrite(os.path.join(save_dir, image_name), label)
