
import argparse
import logging
import os
import random
import sys
import time
import numpy as np
from medpy import metric
import os
# os.environ['CUDA_ENABLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import ndimage
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, SubsetRandomSampler

from typing import List, cast, Tuple, Optional
from torch import Tensor, einsum
from utils import simplex, probs2one_hot
from utils import one_hot2hd_dist


from utils1 import  lovasz_softmax

from networks.TransUnet import VisionTransformer
from networks.TransUnet import CONFIGS

from networks.EfficientUnet import Efficientunet_b4
from networks.deeplabv3_resnet101 import deeplabv3_resnet5zero,deeplabv3_resnet10one,fcn_resnet5zero
from networks.network1 import modeling


from tqdm import tqdm
# from utils import DiceLoss
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
# from utils import DiceLoss




device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def compute_dice_coef_val(input, target):
    '''
    Compute dice score metric.
    '''
    batch_size = input.shape[0]
    # input = input.cpu().detach().numpy()
    # target = target.cpu().detach().numpy()

    return sum([1.0 - metric.binary.dc(input[k, :, :], target[k, :, :])for k in range(batch_size)])


def dice_coeff(input, target):
    '''
    in tensor fomate
    :param input:
    :param target:
    :return:
    '''
    smooth = 1.
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        nn.CrossEntropyLoss()

        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        # print(self.n_classes)
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # print(target.shape)
        target = self._one_hot_encoder(target)
        # print(target.shape)
        if weight is None:
            weight = [1] * self.n_classes
        # pri
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def l1_loss(x, y, reduction='mean'):
    # dif = np.abs(x - y)
    # if reduction == 'mean':
    #     return np.mean(dif)
    # elif reduction == 'sum':
    #     return np.sum(dif)
    # return dif
    # print(x.shape,y.shape)
    batch_size =  x.shape[0]
    # input = input.cpu().detach().numpy()
    # target = target.cpu().detach().numpy()

    return sum([np.mean(abs(x[k, :, :]- y[k, :, :])) for k in range(batch_size)])


def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    device = label.device
    one_hot_label = torch.eye(
        n_classes, device=device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label




def trainer_OCT_goals(args, model, snapshot_path):

    from datasets.dataset_oct_goals import OCTDataset,RandomGeneratorROI,OCTDataset_noargument
    # from datasets.dataset_fovea_localization_disc_up import fovea_dataset,RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train = OCTDataset(base_dir=args.root_path, split="training",
                                    transform=transforms.Compose(
                                        [RandomGeneratorROI(output_size=[args.img_size, args.img_size])]))
    db_train1 = OCTDataset_noargument(base_dir=args.root_path, split="training",
                          transform=transforms.Compose(
                              [RandomGeneratorROI(output_size=[args.img_size, args.img_size])]))
    # db_val = OCTDataset(base_dir=args.root_path, split="val",
    #                                 transform=transforms.Compose(
    #                                     [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, random_state=None, shuffle=True)  # 5折

    lendb = len(db_train)
    lendb1 = int(lendb / 5)
    lendb2 = lendb1 * 2
    lendb3 = lendb1 * 3
    lendb4 = lendb1 * 4

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(db_train)))):  ###五折交叉验证

        print('Fold {}'.format(fold + 1))
        if fold + 1 == 1:
            train_idx = list(range(lendb1, lendb))
            val_idx = list(range(0, lendb1))
        if fold + 1 == 2:
            train_idx = list(range(0, lendb1)) + list(range(lendb2, lendb))
            val_idx = list(range(lendb1, lendb2))
        if fold + 1 == 3:
            train_idx = list(range(0, lendb2)) + list(range(lendb3, lendb))
            val_idx = list(range(lendb2, lendb3))
        if fold + 1 == 4:
            train_idx = list(range(0, lendb3)) + list(range(lendb4, lendb))
            val_idx = list(range(lendb3, lendb4))
        if fold + 1 == 5:
            train_idx = list(range(0, lendb4))
            val_idx = list(range(lendb4, lendb))

        # print(train_idx)
        # print(val_idx)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        print(len(train_sampler))
        print(len(val_sampler))

        # model = Efficientunet_b4(num_classes=4)
        # model = deeplabv3_resnet10one(num_classes=4)
        model =  modeling.__dict__['deeplabv3plus_resnet50'](num_classes=4)


        if args.n_gpu > 1:
            model = nn.DataParallel(model)

        trainloader = DataLoader(db_train, batch_size=batch_size, num_workers=4, pin_memory=False,
                                 worker_init_fn=worker_init_fn, drop_last=False, sampler=train_sampler)
        valloader = DataLoader(db_train1, batch_size=batch_size, num_workers=1, pin_memory=False,
                               worker_init_fn=worker_init_fn, drop_last=False,sampler=val_sampler)

        ce_loss = CrossEntropyLoss()
        # mseloss = nn.MSELoss()
        dice_loss = DiceLoss(num_classes)

        # optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

        torch.autograd.set_detect_anomaly(True)

        writer = SummaryWriter(snapshot_path + '/log')
        iter_num = 0

        min_loss1 = float('inf')
        max_epoch = 50
        max_iterations = max_epoch * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
        logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
        iterator = tqdm(range(max_epoch), ncols=70)

        model.to(device)
        for epoch_num in iterator:

            model.train()
            for i_batch, sampled_batch in enumerate(trainloader):
                # print(ert)
                image_batch, label_batch = sampled_batch['oct_image'], sampled_batch['oct_mask']
                case_name = sampled_batch['case_name']
                # print(case_name)

                image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                B, C, H, W = image_batch.shape

                outputs = model(image_batch)
                # outputs = outputs['out']

                loss_ce = ce_loss(outputs, label_batch[:].long())

                loss_dice = dice_loss(outputs, label_batch, softmax=True)

                loss_total = 0.4 * loss_ce + 0.6 * loss_dice

                loss = loss_total

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1
                # writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)

                logging.info('iteration %d :  loss_dice: %f  loss_ce %f'
                             % (iter_num, loss_dice.item(), loss_ce.item()))

            loss_val = 0.0
            loss_RNFLs = 0.0
            loss_GCIPLs = 0.0
            loss_Choroids = 0.0

            loss1_RNFLs = 0.0
            loss1_GCIPLs = 0.0
            loss1_Choroids = 0.0

            l1loss = nn.L1Loss()

            model.eval()
            with torch.no_grad():
                for i_batch, sampled_batch in enumerate(valloader):
                    image_batch, label_batch = sampled_batch['oct_image'], sampled_batch['oct_mask']
                    RNFL_batch, GCIPL_batch, Choriod_batch = sampled_batch['oct_RNFL'], \
                                                             sampled_batch['oct_GCIPL'], sampled_batch['oct_Choriod']
                    case_name = sampled_batch['case_name']
                    # print(case_name)

                    image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                    B, C, H, W = image_batch.shape
                    # print(image_batch.shape)

                    outputs = model(image_batch)
                    # outputs = outputs['out']

                    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)

                    outputs = outputs.cpu().detach().numpy()
                    RNFL_batch = RNFL_batch.cpu().detach().numpy()
                    GCIPL_batch = GCIPL_batch.cpu().detach().numpy()
                    Choriod_batch = Choriod_batch.cpu().detach().numpy()

                    outputs3_RNFL = np.zeros_like(outputs)
                    outputs3_GCIPL = np.zeros_like(outputs)
                    outputs3_Choriod = np.zeros_like(outputs)

                    outputs3_RNFL[outputs == 0] = 1.0
                    outputs3_GCIPL[outputs == 1] = 1.0
                    outputs3_Choriod[outputs == 2] = 1.0



                    loss_RNFL = compute_dice_coef_val(outputs3_RNFL, RNFL_batch)
                    loss_RNFLs += loss_RNFL

                    loss_GCIPL = compute_dice_coef_val(outputs3_GCIPL, GCIPL_batch)
                    loss_GCIPLs += loss_GCIPL

                    loss_Choroid = compute_dice_coef_val(outputs3_Choriod, Choriod_batch)
                    loss_Choroids += loss_Choroid



                    loss1_RNFL = l1_loss(outputs3_RNFL, RNFL_batch)
                    loss1_RNFLs += loss1_RNFL

                    loss1_GCIPL = l1_loss(outputs3_GCIPL, GCIPL_batch)
                    loss1_GCIPLs += loss1_GCIPL

                    loss1_Choroid = l1_loss(outputs3_Choriod, Choriod_batch)
                    loss1_Choroids += loss1_Choroid



                    seg_err = 0.33 * loss_RNFL + 0.33 * loss_GCIPL + 0.33 * loss_Choroid


                    l1loss = 0.33 * loss1_GCIPL + 0.33 * loss1_RNFL + 0.33 * loss1_Choroid

                    # loss_val = loss_val + seg_err

                    loss_val = loss_val + seg_err +  10 * l1loss

                    # print(seg_err/B, l1loss/B)

                logging.info('iteration %d : loss_val : %f min_loss1 : %f '
                             % (iter_num, loss_val / len(val_sampler), min_loss1 / len(val_sampler)))


                logging.info('iteration %d : dice_RNFL : %f dice_GCIPL : %f  dice_Choroid : %f'
                             % (iter_num, 1.0 - loss_RNFLs / len(val_sampler),
                                1.0 - loss_GCIPLs / len(val_sampler), 1.0 - loss_Choroids / len(val_sampler)))

                logging.info('iteration %d : loss1_RNFL : %f loss1_GCIPL : %f  loss1_Choroid : %f'
                             % (iter_num,  loss1_RNFLs / len(val_sampler),
                                 loss1_GCIPLs / len(val_sampler),  loss1_Choroids / len(val_sampler)))

                if loss_val < min_loss1:
                    min_loss1 = loss_val
                    save_mode_path = os.path.join(snapshot_path, 'best_model' + str(fold + 1) + '.pth')
                    torch.save(model.state_dict(), save_mode_path)

                # if epoch_num >= max_epoch - 1:
                #     save_mode_path = os.path.join(snapshot_path, 'best_model' + str(fold + 1) + '.pth')
                #     torch.save(model.state_dict(), save_mode_path)

        writer.close()
    return "Training Finished!"


def trainer_OCT_goals_nocross(args, model, snapshot_path):

    from datasets.dataset_oct_goals import OCTDataset,RandomGeneratorROI
    # from datasets.dataset_fovea_localization_disc_up import fovea_dataset,RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    temperature: float = 1


    db_train = OCTDataset(base_dir=args.root_path, split="training",
                                    transform=transforms.Compose(
                                        [RandomGeneratorROI(output_size=[args.img_size, args.img_size])]))
    # db_val = OCTDataset(base_dir=args.root_path, split="val",
    #                                 transform=transforms.Compose(
    #                                     [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    trainloader = DataLoader(db_train, batch_size=batch_size, num_workers=4,shuffle=True, pin_memory=False,
                             worker_init_fn=worker_init_fn, drop_last=False)
    # valloader = DataLoader(db_train, batch_size=batch_size, num_workers=1, pin_memory=False,
    #                        worker_init_fn=worker_init_fn, drop_last=False, sampler=val_sampler)

    ce_loss = CrossEntropyLoss()
    mseloss = nn.MSELoss()
    dice_loss = DiceLoss(num_classes)
    list1 = [0.2,0.2,0.2,0.4]
    focal_loss = FocalLoss()
    # focal_loss = FocalLoss()
    # dice_loss_edge = DiceLoss(2)
    # lovasz_loss = lovasz_softmax()

    # optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    torch.autograd.set_detect_anomaly(True)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0

    min_loss1 = float('inf')
    max_epoch = 600
    max_iterations = max_epoch * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)

    model.to(device)
    for epoch_num in iterator:

        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['oct_image'], sampled_batch['oct_mask']
            RNFL_batch, GCIPL_batch, Choriod_batch = sampled_batch['oct_RNFL'], \
                                                     sampled_batch['oct_GCIPL'], sampled_batch['oct_Choriod']
            case_name = sampled_batch['case_name']


            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            RNFL_batch, GCIPL_batch,Choriod_batch = RNFL_batch.to(device), GCIPL_batch.to(device),Choriod_batch.to(device)

            # edge_batch = edge_batch.to(device)

            B, C, H, W = image_batch.shape


            outputs = model(image_batch)
            # outputs = outputs['out']


            # predicted_mask: Tensor = probs2one_hot(pred_probs.detach())  # Used only for dice computation

            # print(outputs.shape,label_batch.shape)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            # loss_ce = focal_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # outputs = F.softmax(outputs, dim=1)
            # loss_lovasz_softmax = lovasz_softmax(outputs, label_batch)
            # loss_edge =  dice_loss_edge(outputs_edge, edge_batch,softmax=True)

            loss_total = 0.4 * loss_ce + 0.6 * loss_dice

            # loss_total = 0.5 * loss_ce + 0.5 * loss_lovasz_softmax

            # loss_total =  0.3*(0.4 * loss_ce + 0.6 * loss_dice) + 0.7 * loss_boundry
            # loss_total = 0.4 * loss_boundry + 0.6 * loss_dice
            # loss_total = (0.4 * loss_ce + 0.6 * loss_dice)*0.7 + 0.3*loss_boundry

            # RNFL = torch.sigmoid(outputs[:, 0, :, :].to(torch.float32))
            # GCIPL = torch.sigmoid(outputs[:, 1, :, :].to(torch.float32))
            # Choriod = torch.sigmoid(outputs[:, 2, :, :].to(torch.float32))
            #
            # loss_RNFL = dice_coeff(RNFL,RNFL_batch)
            # loss_GCIPL = dice_coeff(GCIPL,GCIPL_batch)
            # loss_Choriod = dice_coeff(Choriod,Choriod_batch)
            #
            # loss_total = loss_RNFL + loss_GCIPL + loss_Choriod


            loss = loss_total
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_


            iter_num = iter_num + 1
            # writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            logging.info('iteration %d :  loss_dice: %f  loss_ce %f'
                         % (iter_num, loss_dice.item(), loss_ce.item()))

            # logging.info('iteration %d :  loss_dice: %f  loss_ce %f'
            #              % (iter_num, loss_lovasz_softmax.item(), loss_ce.item()))

            # logging.info('iteration %d :  loss_RNFL: %f  loss_GCIPL %f loss_Choriod %f'
            #              % (iter_num, 1.0 -loss_RNFL.item(), 1.0 - loss_GCIPL.item(), 1.0-loss_Choriod.item()))

            # logging.info('iteration %d :  loss_dice: %f  loss_focal %f  loss_edge: %f'
            #              % (iter_num, loss_dice.item(), loss_ce.item(),loss_boundry.item()))

        save_interval = 50  # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num  == 50:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))


        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_num'  + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))


    # from sklearn.model_selection import KFold
    #
    # kf = KFold(n_splits=5, random_state=None, shuffle=True)  # 5折
    #
    # lendb = len(db_train)
    # lendb1 = int(lendb / 5)
    # lendb2 = lendb1 * 2
    # lendb3 = lendb1 * 3
    # lendb4 = lendb1 * 4
    #
    # for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(db_train)))):  ###五折交叉验证
    #
    #     print('Fold {}'.format(fold + 1))
    #     if fold + 1 == 1:
    #         train_idx = list(range(lendb1, lendb))
    #         val_idx = list(range(0, lendb1))
    #     if fold + 1 == 2:
    #         train_idx = list(range(0, lendb1)) + list(range(lendb2, lendb))
    #         val_idx = list(range(lendb1, lendb2))
    #     if fold + 1 == 3:
    #         train_idx = list(range(0, lendb2)) + list(range(lendb3, lendb))
    #         val_idx = list(range(lendb2, lendb3))
    #     if fold + 1 == 4:
    #         train_idx = list(range(0, lendb3)) + list(range(lendb4, lendb))
    #         val_idx = list(range(lendb3, lendb4))
    #     if fold + 1 == 5:
    #         train_idx = list(range(0, lendb4))
    #         val_idx = list(range(lendb4, lendb))
    #
    #     # print(train_idx)
    #     # print(val_idx)
    #     train_sampler = SubsetRandomSampler(train_idx)
    #     val_sampler = SubsetRandomSampler(val_idx)
    #     print(len(train_sampler))
    #     print(len(val_sampler))
    #
    #     config_vit = CONFIGS['R50-ViT-B_16']
    #     config_vit.n_skip = 3
    #     config_vit.n_classes = args.num_classes
    #
    #     config_vit.patches.grid = (int(224 / 16), int(448 / 16))
    #
    #     model = VisionTransformer(config_vit, img_size=[224, 448], num_classes=args.num_classes)
    #
    #     if args.n_gpu > 1:
    #         model = nn.DataParallel(model)
    #
    #     trainloader = DataLoader(db_train, batch_size=batch_size, num_workers=4, pin_memory=False,
    #                              worker_init_fn=worker_init_fn, drop_last=False, sampler=train_sampler)
    #     valloader = DataLoader(db_train, batch_size=batch_size, num_workers=1, pin_memory=False,
    #                            worker_init_fn=worker_init_fn, drop_last=False,sampler=val_sampler)
    #
    #     ce_loss = CrossEntropyLoss()
    #     # mseloss = nn.MSELoss()
    #     dice_loss = DiceLoss(num_classes)
    #
    #     optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    #     # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    #
    #     torch.autograd.set_detect_anomaly(True)
    #
    #     writer = SummaryWriter(snapshot_path + '/log')
    #     iter_num = 0
    #
    #     min_loss1 = float('inf')
    #     max_epoch = 200
    #     max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    #     logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    #     iterator = tqdm(range(max_epoch), ncols=70)
    #
    #     model.to(device)
    #     for epoch_num in iterator:
    #
    #         model.train()
    #         for i_batch, sampled_batch in enumerate(trainloader):
    #             # print(ert)
    #             image_batch, label_batch = sampled_batch['oct_image'], sampled_batch['oct_mask']
    #             case_name = sampled_batch['case_name']
    #             # print(case_name)
    #
    #             image_batch, label_batch = image_batch.to(device), label_batch.to(device)
    #             B, C, H, W = image_batch.shape
    #             # print(image_batch.shape)
    #
    #             r = random.uniform(0.0, 1.0)
    #             stddev = random.uniform(0.001, 0.005)
    #             # noise = torch.normal(0, std=stddev, dtype=torch.float32,out=[32, 3, 224, 224],)
    #             noise = torch.normal(mean=torch.full((B, 3, H, W), 0.0), std=torch.full((B, 3, H, W), stddev)).to(
    #                 device)
    #             # print(noise)
    #             if r > 0.6:
    #                 image_batch = image_batch + noise
    #
    #             outputs, side_6, side_7 = model(image_batch)
    #
    #             loss_ce = ce_loss(outputs, label_batch[:].long())
    #             loss_dice = dice_loss(outputs, label_batch, softmax=True)
    #
    #             loss_total = 0.4 * loss_ce + 0.6 * loss_dice
    #
    #             loss = loss_total
    #
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #
    #             iter_num = iter_num + 1
    #             # writer.add_scalar('info/lr', lr_, iter_num)
    #             writer.add_scalar('info/total_loss', loss, iter_num)
    #
    #             logging.info('iteration %d :  loss3_2: %f  '
    #                          % (iter_num, loss_dice.item(),))
    #
    #         loss_val = 0.0
    #         loss_RNFLs = 0.0
    #         loss_GCIPLs = 0.0
    #         loss_Choroids = 0.0
    #
    #         model.eval()
    #         with torch.no_grad():
    #             for i_batch, sampled_batch in enumerate(valloader):
    #                 image_batch, label_batch = sampled_batch['oct_image'], sampled_batch['oct_mask']
    #                 RNFL_batch, GCIPL_batch, Choriod_batch = sampled_batch['oct_RNFL'], \
    #                                                          sampled_batch['oct_GCIPL'], sampled_batch['oct_Choriod']
    #                 case_name = sampled_batch['case_name']
    #                 # print(case_name)
    #
    #                 image_batch, label_batch = image_batch.to(device), label_batch.to(device)
    #                 B, C, H, W = image_batch.shape
    #                 # print(image_batch.shape)
    #
    #                 outputs, side_6, side_7 = model(image_batch)
    #
    #                 outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
    #
    #                 outputs = outputs.cpu().detach().numpy()
    #                 RNFL_batch = RNFL_batch.cpu().detach().numpy()
    #                 GCIPL_batch = GCIPL_batch.cpu().detach().numpy()
    #                 Choriod_batch = Choriod_batch.cpu().detach().numpy()
    #
    #                 outputs3_RNFL = np.zeros_like(outputs)
    #                 outputs3_GCIPL = np.zeros_like(outputs)
    #                 outputs3_Choriod = np.zeros_like(outputs)
    #
    #                 outputs3_RNFL[outputs == 0] = 1.0
    #                 outputs3_GCIPL[outputs == 1] = 1.0
    #                 outputs3_Choriod[outputs == 2] = 1.0
    #
    #                 loss_RNFL = compute_dice_coef_val(outputs3_RNFL, RNFL_batch)
    #                 loss_RNFLs += loss_RNFL
    #
    #                 loss_GCIPL = compute_dice_coef_val(outputs3_GCIPL, GCIPL_batch)
    #                 loss_GCIPLs += loss_GCIPL
    #
    #                 loss_Choroid = compute_dice_coef_val(outputs3_Choriod, Choriod_batch)
    #                 loss_Choroids += loss_Choroid
    #
    #                 seg_err = 0.4 * loss_RNFL + 0.3 * loss_GCIPL + 0.3 * loss_Choroid
    #
    #                 loss_val = loss_val + seg_err
    #                 # loss_val = loss_val + fovea_err*len(image_batch)
    #
    #             logging.info('iteration %d : loss_val : %f min_loss1 : %f '
    #                          % (iter_num, loss_val / len(val_sampler), min_loss1 / len(val_sampler)))
    #
    #             logging.info('iteration %d : dice_RNFL : %f dice_GCIPL : %f  dice_Choroid : %f'
    #                          % (iter_num, 1.0 - loss_RNFLs / len(val_sampler),
    #                             1.0 - loss_GCIPLs / len(val_sampler), 1.0 - loss_Choroids / len(val_sampler)))
    #
    #             if loss_val < min_loss1:
    #                 min_loss1 = loss_val
    #
    #             if epoch_num >= max_epoch - 1:
    #                 save_mode_path = os.path.join(snapshot_path, 'best_model' + str(fold + 1) + '.pth')
    #                 torch.save(model.state_dict(), save_mode_path)

    writer.close()
    return "Training Finished!"











