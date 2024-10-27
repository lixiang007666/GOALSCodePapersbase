import paddlex as pdx
from paddlex import transforms as T


def main(fold_num: int):
    train_transforms = T.Compose([
        T.RandomHorizontalFlip(prob=0.5),
        T.RandomCrop(crop_size=224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_transforms = T.Compose([
        T.ResizeByShort(short_size=256),
        T.CenterCrop(crop_size=224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = pdx.datasets.ImageNet(
        data_dir='GOALS2022-Train/Train',
        file_list=f'split_lists/cls_cv/train_{fold_num}.txt',
        label_list='split_lists/cls_cv/labels.txt',
        transforms=train_transforms,
        shuffle=True)
    eval_dataset = pdx.datasets.ImageNet(
        data_dir='GOALS2022-Train/Train',
        file_list=f'split_lists/cls_cv/val_{fold_num}.txt',
        label_list='split_lists/cls_cv/labels.txt',
        transforms=eval_transforms)

    model = pdx.cls.PPLCNet(num_classes=len(train_dataset.labels), scale=1.)

    batch_size = 10

    model.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

        num_epochs=50,
        train_batch_size=batch_size,
        learning_rate=0.0005,

        warmup_steps=(train_dataset.num_samples // batch_size) * 5,
        warmup_start_lr=0.0,

        save_interval_epochs=5,
        log_interval_steps=train_dataset.num_samples // batch_size,
        save_dir=f'models_cls/PPLCNet_Fold{fold_num}',
        pretrain_weights='IMAGENET',
        use_vdl=True)

if __name__ == '__main__':
    for fold_num in range(1, 11):
        main(fold_num=fold_num)
