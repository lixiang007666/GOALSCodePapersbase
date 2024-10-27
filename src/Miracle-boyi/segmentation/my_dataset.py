import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "Train" if train else "test"
        data_root = os.path.join(root, "data", self.flag)   # ./data/Train
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "Image")) if i.endswith(".png")]
        self.img_list = [os.path.join(data_root, "Image", i) for i in img_names]
        self.manual = [os.path.join(data_root, "Layer_Masks", i)
                       for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        manual = Image.open(self.manual[idx]).convert('L')
        # 前景变为1，背景变为0
        manual = np.array(manual)

        # 预测第一个类-----0
        # manual[manual == 0] = 1
        # manual[manual == 80] = 0
        # manual[manual == 160] = 0
        # manual[manual == 255] = 0

        # 预测第二个类-----80
        manual[manual == 0] = 0
        manual[manual == 80] = 1
        manual[manual == 160] = 0
        manual[manual == 255] = 0

        # 预测第三个类-----160
        # manual[manual == 0] = 0
        # manual[manual == 80] = 0
        # manual[manual == 160] = 1
        # manual[manual == 255] = 0


        mask = np.clip(manual, a_min=0, a_max=255)
        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask

    def __len__(self):
        return len(self.img_list)

    # 对数据进行batch的打包处理
    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

