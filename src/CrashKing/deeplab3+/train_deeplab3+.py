import argparse
import logging
import os
import numpy as np
from pathlib import Path
import torch
from matplotlib import pyplot as plt
from torch import optim, einsum
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from modeling.deeplab import DeepLab
from utils.data_loading import OCTDataset
from evaluate import evaluate

from losses import CELoss


def plot_loss(y, name, scale=1):
    x = list(n * scale for n in range(0, len(y)))
    plt.figure('dice_loss')
    plt.subplot(1, 1, 1)
    plt.plot(x, y, '.-')
    plt.ylabel('LOSS')
    plt.savefig(name)


# # --------------------------- BINARY LOSSES ---------------------------
from scipy.ndimage import distance_transform_edt as distance


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    # assert one_hot(torch.Tensor(seg), axis=0)
    seg = np.array(seg.cpu())
    N: int = seg.shape[0]
    C: int = seg.shape[1]

    res = np.zeros_like(seg)
    for n in range(N):
        for c in range(C):
            posmask = seg[n][c].astype(np.bool)

            if posmask.any():
                negmask = ~posmask
                # print('negmask:', negmask)
                # print('distance(negmask):', distance(negmask))
                res[n][c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
                # print('res[c]', res[c])
    return res


class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = [0, 1, 2]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs, dist_maps):
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss


BoundaryLoss = SurfaceLoss


def train_net(
        args,
        net,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        amp: bool = False):
    dir_img = args.dir_img
    dir_mask = args.dir_mask
    dir_checkpoint = args.dir_checkpoint + '_' + args.model.lower() + '/'

    dataset = OCTDataset(dir_img, dir_mask, new_size=args.output_size)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)

    global_step = 0
    best_val_score = 0

    ce_loss = CELoss()
    # boundaryloss
    soft = SurfaceLoss()

    # define loss function
    val_dice_loss_list_plot = []
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                mask2 = batch['mask2']
                assert images.shape[1] == net.in_channels, \
                    f'Network has been defined with {net.in_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                # Get distance
                seg2 = one_hot2dist(mask2)
                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    celoss = ce_loss(masks_pred, true_masks)
                    # Boundaryloss
                    soft_loss = soft(torch.softmax(masks_pred, dim=1), torch.Tensor(seg2).to(device))
                    a = 0.4
                    loss = a * celoss + (1 - a) * soft_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))

                        val_dice_loss_list_plot.append(val_score.cpu())
                        # plot_loss(train_avg_dice_list_plot, name='train_avg_dice_list_plot_{}.jpg'.format(args.model.lower()))
                        plot_loss(val_dice_loss_list_plot, scale=division_step,
                                  name='val_dice_loss_list_plot_{}.jpg'.format(args.model.lower()))

                        # save best model
                        if best_val_score < val_score:
                            if best_val_score != 0:
                                del_path = dir_checkpoint + 'best_checkpoint_dice{:.5f}.pth'.format(best_val_score)
                                os.remove(del_path)  # 递归删除文件夹
                            best_val_score = val_score
                            save_path = dir_checkpoint + 'best_checkpoint_dice{:.5f}.pth'.format(best_val_score)
                            if os.path.exists(save_path):
                                os.remove(save_path)  # 递归删除文件夹
                            if save_checkpoint:
                                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                                torch.save(net.state_dict(),
                                           dir_checkpoint + '/best_checkpoint_dice{:.5f}.pth'.format(best_val_score))
                                logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the Model on Glaucoma Oct Analysis and Layer Segmentation task')
    parser.add_argument('--model', '-m', metavar='M', type=str, default='deeplab', help='Name of model')

    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--val_ratio', '-v', metavar='V', type=float, default=0.2,
                        help='Percent of the data that is used as validation')
    parser.add_argument('--output_size', type=int, default=512, help='Output size of the images')

    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')

    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=4, help='Number of classes')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')

    parser.add_argument('--device', default='0', type=str)

    parser.add_argument('--dir_img', type=str, default='./preprocess_train_data/image', help='images directory')
    parser.add_argument('--dir_mask', type=str, default='./preprocess_train_data/label', help='masks directory')
    parser.add_argument('--dir_checkpoint', type=str, default='./checkpoints', help='model saved directory')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Using device {device}')

    if args.model == 'deeplab':
        # Define network
        net = DeepLab(num_classes=4,
                      backbone='resnet')
    else:
        raise NotImplementedError

    logging.info(f'Network:\n'
                 f'\t{net.in_channels} input channels\n'
                 f'\t{net.num_classes} output channels (classes)\n')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(
            args=args,
            net=net,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            val_percent=args.val_ratio,
            amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
