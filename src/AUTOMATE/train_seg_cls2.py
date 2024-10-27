import paddle
import paddleseg
from paddleseg import transforms as T
from paddleseg.core import train
from paddleseg.models import DiceLoss


def main(fold_num: int, num_classes: int):
    train_transforms = [
        T.RandomRotation(max_rotation=20),
        T.RandomPaddingCrop(crop_size=(1100, 800)),
        T.Padding(target_size=(1120, 800)),
        T.RandomHorizontalFlip(prob=0.5),
        T.RandomDistort(
            brightness_range=0.3, brightness_prob=0.5,
            contrast_range=0.3, contrast_prob=0.5,
            saturation_range=0.3, saturation_prob=0.5,
            hue_range=0, hue_prob=0,
            sharpness_range=0, sharpness_prob=0),
        T.RandomBlur(prob=0.1, blur_type='random'),
        T.Normalize(
            mean=[0.2297999174664977, 0.2297999174664977, 0.2297999174664977],
            std=[0.16316278003205756, 0.16316278003205756, 0.16316278003205756])
    ]
    eval_transforms = [
        T.Padding(target_size=(1120, 800)),
        T.Normalize(
            mean=[0.2297999174664977, 0.2297999174664977, 0.2297999174664977],
            std=[0.16316278003205756, 0.16316278003205756, 0.16316278003205756])
    ]

    train_dataset = paddleseg.datasets.Dataset(
        mode='train',
        num_classes=num_classes,
        dataset_root='GOALS2022-Train_cls2/Train',
        train_path=f'split_lists/seg_cv/train_{fold_num}.txt',
        transforms=train_transforms,
        edge=False)
    eval_dataset = paddleseg.datasets.Dataset(
        mode='val',
        num_classes=num_classes,
        dataset_root='GOALS2022-Train_cls2/Train',
        val_path=f'split_lists/seg_cv/val_{fold_num}.txt',
        transforms=eval_transforms)

    model = paddleseg.models.U2Net(num_classes=num_classes)

    iters = 4000
    train_batch_size = 2
    learning_rate = 0.0005
    eta_min = 0.000001

    decayed_lr = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=learning_rate,
        T_max=iters,
        eta_min=eta_min)

    decayed_lr = paddle.optimizer.lr.LinearWarmup(
        learning_rate=decayed_lr,
        warmup_steps=200,
        start_lr=eta_min,
        end_lr=learning_rate)

    optimizer = paddle.optimizer.AdamW(
        learning_rate=decayed_lr,
        parameters=model.parameters(),
        weight_decay=0.01)

    losses = {
        'types': [DiceLoss()] * 7,
        'coef': [1] * 7
    }

    train(
        train_dataset=train_dataset,
        val_dataset=eval_dataset,

        model=model,
        optimizer=optimizer,
        losses=losses,

        iters=iters,
        batch_size=train_batch_size,

        save_interval=160,
        log_iters=10,
        save_dir=f'models_seg/cls2_{type(model).__name__}_Dice_Norm_Fold{fold_num}',
        use_vdl=True,
        keep_checkpoint_max=1)


if __name__ == '__main__':
    for i in range(1, 6):
        main(fold_num=i, num_classes=2)
