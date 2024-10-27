import torch
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import datetime
from model.gatenet_rgb_res50_pami import GateNet, GateNet_dem_128,GateNet_dem_512,GateNet_dem_128_convnext,GateNet_dem_512_convnext,GateNet_dem_512_tf_efficientnet_b7_ns,GateNet_dem_512_resnest50d_4s2x40d
from model.miccai_msnet import MSNet,MSNet_128,MSNet_pvtv2
from model.HEL import HEL
from model.hfloss import HausdorffDTLoss
from model.minet import MINet_Resnet,MINet_Resnet_dem128
from utils.dataset_rgb_strategy2_w_h import get_loader
from utils.utils_segmentation import adjust_lr, AvgMeter
import torch.nn as nn
from torch.cuda import amp


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--trainsize_h', type=int, default=880, help='training dataset size')
# parser.add_argument('--trainsize_h', type=int, default=352, help='training dataset size')
parser.add_argument('--trainsize_w', type=int, default=640, help='training dataset size')
# parser.add_argument('--trainsize_w', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
#100epoch, default=30, 200epoch default=80, 300epoch default=150, 400epoch default=240,500epoch default=300,
# parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
parser.add_argument('-beta1_gen', type=float, default=0.5,help='beta of Adam for generator')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--feat_channel', type=int, default=64, help='reduced channel of saliency feat')

opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
generator = MSNet()
generator.cuda()
generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)

## load data
image_root = '/home/asus/Datasets/GOALS_challenges_miccai2022/GOALS2022-Train/Train/Image/'
gt_root = '/home/asus/Datasets/GOALS_challenges_miccai2022/GOALS2022-Train/Train/mask_3/'


train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize_w=opt.trainsize_w,
                          trainsize_h=opt.trainsize_h)
total_step = len(train_loader)

## define loss

CE = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
size_rates = [0.75,1,1.25]  # multi-scale training
# size_rates = [0.5,0.75,1,1.25,1.5]  # multi-scale training
# size_rates = [1]  # multi-scale training
criterion = nn.BCEWithLogitsLoss().cuda()
criterion_mae = nn.L1Loss().cuda()
criterion_mse = nn.MSELoss().cuda()
use_fp16 = True
scaler = amp.GradScaler(enabled=use_fp16)
HEL_loss = HEL()
HF_loss = HausdorffDTLoss()
def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))


    pred  = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou  = 1-(inter+1)/(union-inter+1)

    return (wbce+wiou).mean()

def visualize_mi_rgb(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_rgb_mi.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_mi_depth(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_depth_mi.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

## visualize predictions and gt
def visualize_rgb_init(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_rgb_int.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_depth_init(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_depth_int.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_rgb_ref(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_rgb_ref.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_depth_ref(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_depth_ref.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_final_rgbd(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_rgbd.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_uncertainty_prior_init(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_prior_int.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

## linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed

print("Let's Play!")
save_path = './saved_model/GOALS_challenges_miccai2022_mask_3_MSNet_with_foldaspp_gap_resnext101_32x8d_datasets_rgb_strategy2_w_h_733_534_size_ms_5scale_batch4_train_100epoch_add_all_dataenhanced'
if not os.path.exists(save_path):
    os.makedirs(save_path)
log_path = os.path.join(save_path, str(datetime.datetime.now()) + '.txt')
open(log_path, 'w')
for epoch in range(1, opt.epoch+1):
    generator.train()
    loss_record = AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            images, gts = pack
            images = Variable(images)
            gts = Variable(gts)
            images = images.cuda()
            gts = gts.cuda()
            # multi-scale training samples
            # multi-scale training samples
            # multi-scale training samples
            trainsize_h = int(round(opt.trainsize_h * rate / 1) * 1)
            trainsize_w = int(round(opt.trainsize_w * rate / 1) * 1)
            if rate != 1:
                images = F.upsample(images, size=(trainsize_w, trainsize_h), mode='bilinear',
                                    align_corners=True)
                gts = F.upsample(gts, size=(trainsize_w, trainsize_h), mode='bilinear', align_corners=True)

            b, c, h, w = gts.size()
            target_1 = F.upsample(gts, size=h // 2, mode='nearest')
            target_2 = F.upsample(gts,  size=h // 4, mode='nearest')

            with amp.autocast(enabled=use_fp16):
                # output_fpn, output_final = generator.forward(images)  # hed
                output_final = generator.forward(images)  # hed
                # loss1 = structure_loss(output_fpn, gts)
                # loss1 = HEL_loss(output_fpn, gts)
                # loss2 = HEL_loss(output_final, gts)
                # loss1 = HEL_loss(F.sigmoid(output_final),gts)
                # loss1 = HF_loss(F.sigmoid(output_final),gts)
                loss2 = structure_loss(output_final, gts)

                loss = loss2

            generator_optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(generator_optimizer)
            scaler.update()

            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)


        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}'.
                  format(datetime.datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))

            log = ('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}'.
                  format(datetime.datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))
            open(log_path, 'a').write(log + '\n')
            # print(anneal_reg)


    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
    if epoch % opt.epoch == 0:
        torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
