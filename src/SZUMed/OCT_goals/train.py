import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision

from networks.TransUnet import VisionTransformer
from networks.TransUnet import CONFIGS

from networks.deeplabv3_resnet101 import deeplabv3_resnet5zero,deeplabv3_resnet10one,fcn_resnet5zero
from networks.R50_U_Net import UNetWithResnet50Encoder
from networks.EfficientUnet import Efficientunet_b4

from networks.segnet import  SegNet

from networks.network1 import modeling

# from networks.deeplabv3_resnet101 import deeplabv3_resnet10one

from trainer import trainer_OCT_goals,trainer_OCT_goals_nocross

from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()

config = get_config(args)


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset

    list_ablation = ['transunet','deeplabv3','Unet','efficient','fcn']
    args.ablation = list_ablation[1]

    args.output_dir = args.output_dir + '/' + args.ablation
    print(args.output_dir)


    dataset_config = {
        'OCT_goal': {
            'root_path': args.root_path,
            'num_classes': 4,
        },
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24


    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)



    config_vit = CONFIGS['R50-ViT-B_16']
    config_vit.n_skip = 3
    config_vit.n_classes = args.num_classes

    config_vit.patches.grid = ( int(512 / 16), int(1024 / 16) )

    net = VisionTransformer(config_vit, img_size=[512,896], num_classes=args.num_classes)
    # # print(config_vit.pretrained_path)
    # # net.load_from(weights=np.load(config_vit.pretrained_path))
    # net = deeplabv3_resnet34(num_classes=4)
    # net = deeplabv3_resnet5zero(num_classes=4)
    net =  deeplabv3_resnet10one(num_classes=4)
    # net = fcn_resnet5zero(num_classes=4)
    # net = UNetWithResnet50Encoder(n_classes=4)
    # net = Efficientunet_b4(num_classes=4)
    # print(111)

    # net = SegNet(input_nbr=3,label_nbr=4)

    # net = modeling.__dict__['deeplabv3_resnet34'](num_classes=4)


    trainer = {'OCT_goal': trainer_OCT_goals}
    trainer[dataset_name](args, net, args.output_dir)