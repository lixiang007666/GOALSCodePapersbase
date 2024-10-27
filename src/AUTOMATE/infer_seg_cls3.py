import paddleseg
from paddleseg import transforms as T
from paddleseg.core import predict

import glob
import cv2

model = paddleseg.models.FCN(
    num_classes=2,
    backbone=paddleseg.models.backbones.HRNet_W18(),
    backbone_indices=(-1,))

image_path_list = glob.glob('GOALS2022-Validation/GOALS2022-Validation/Image/*.png')

eval_transforms = [
    T.Normalize(mean=[0.22894311, 0.22894311, 0.22894311], std=[0.16314624, 0.16314624, 0.16314624])
]

predict(
    model=model,
    model_path='models_seg/cls3_FCN18_WDice/best_model/model.pdparams',
    transforms=T.Compose(eval_transforms),
    image_list=image_path_list,
    save_dir='data/cls3',
    aug_pred=True,
    flip_horizontal=True,
    flip_vertical=False,
    custom_color=[255,255,255, 160,160,160])

image_path_list = glob.glob('data/cls3/pseudo_color_prediction/*.png')
for image_path in image_path_list:
    label = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(image_path, label)
