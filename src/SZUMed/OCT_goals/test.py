import argparse
import logging
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image
from medpy import metric
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_oct_goals import OCTDataset

from networks.segnet import  SegNet


from networks.TransUnet import VisionTransformer
# from networks.TransUnet import CONFIGS

# from networks.TranUnet_AG_NetROI import  VisionTransformer
# from networks.resnet import resnet34
# from networks.TranUnet_AG_NetROI import CONFIGS

from torchvision import transforms
from scipy.ndimage import zoom

from networks.deeplabv3_resnet101 import deeplabv3_resnet5zero,fcn_resnet5zero,deeplabv3_resnet10one


from networks.R50_U_Net import UNetWithResnet50Encoder
from networks.EfficientUnet import Efficientunet_b4
from networks.network1 import modeling

from config import get_config
from networks.TransUnet import CONFIGS
# from networks.unet_model import UNet
import time

from trainer import DiceLoss


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))


parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/disc_test', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
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
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
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

    list_ablation = ['transunet', 'deeplabv3', 'Unet', 'efficient', 'fcn']
    args.ablation = list_ablation[1]

    args.output_dir = args.output_dir + '/' + args.ablation
    print(args.output_dir)

    dataset_config = {
        'OCT_goal': {
            'Dataset': OCTDataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_fovea',
            'num_classes': 4,
            'z_spacing': 1,
        },

    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # net = resnet34()
    # config_vit = CONFIGS['R50-ViT-B_16']
    # config_vit.n_skip = 3
    # config_vit.n_classes = args.num_classes
    #
    # config_vit.patches.grid = (int(512 / 16), int(1024 / 16))
    #
    # model = VisionTransformer(config_vit, img_size=[512, 1024], num_classes=args.num_classes)


    # model = deeplabv3_resnet5zero(num_classes=4)
    # model = UNetWithResnet50Encoder(n_classes=4)
    # model = Efficientunet_b4(num_classes=4)
    # model = fcn_resnet5zero(num_classes=4)
    # model = deeplabv3_resnet10one(num_classes=4)
    # model = SegNet(input_nbr=3,label_nbr=4)

    # model = modeling.__dict__['deeplabv3plus_resnet50'](num_classes=4)

    # snapshot = os.path.join(args.output_dir, 'best_model.pth')

    # snapshot = os.path.join(args.output_dir, 'epoch_599.pth')
    # # print(snapshot)
    # if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    # print(snapshot)

    # model = nn.DataParallel(model)
    cudnn.benchmark = True


    # msg = model.load_state_dict(torch.load(snapshot))
    # model.to(device)
    # model.eval()

    model_file_list = [
        args.output_dir + '/deeplabv3plus_1/best_model1.pth',
        args.output_dir + '/deeplabv3plus_1/best_model2.pth',
        args.output_dir + '/deeplabv3plus_1/best_model3.pth',
        args.output_dir + '/deeplabv3plus_1/best_model4.pth',
        args.output_dir + '/deeplabv3plus_1/best_model5.pth'
    ]

    model_file_list1 = [
        args.output_dir + '/deeplabv3/best_model1.pth',
        args.output_dir + '/deeplabv3/best_model2.pth',
        args.output_dir + '/deeplabv3/best_model3.pth',
        args.output_dir + '/deeplabv3/best_model4.pth',
        args.output_dir + '/deeplabv3/best_model5.pth'
    ]

    model_file_list2 = [
        args.output_dir + '/deeplabv3plus_2/best_model1.pth',
        args.output_dir + '/deeplabv3plus_2/best_model2.pth',
        args.output_dir + '/deeplabv3plus_2/best_model3.pth',
        args.output_dir + '/deeplabv3plus_2/best_model4.pth',
        args.output_dir + '/deeplabv3plus_2/best_model5.pth'
    ]

    net1 = modeling.__dict__['deeplabv3plus_resnet50'](num_classes=4)
    net2 = modeling.__dict__['deeplabv3plus_resnet50'](num_classes=4)
    net3 = modeling.__dict__['deeplabv3plus_resnet50'](num_classes=4)
    net4 = modeling.__dict__['deeplabv3plus_resnet50'](num_classes=4)
    net5 = modeling.__dict__['deeplabv3plus_resnet50'](num_classes=4)

    # net1 =  deeplabv3_resnet5zero(4)
    net1.load_state_dict(torch.load(model_file_list[0]))
    net1.to(device)
    net1.eval()

    # net2 =    deeplabv3_resnet5zero(4)
    net2.load_state_dict(torch.load(model_file_list[1]))
    net2.to(device)
    net2.eval()

    # net3 =   deeplabv3_resnet5zero(4)
    net3.load_state_dict(torch.load(model_file_list[2]))
    net3.to(device)
    net3.eval()

    # net4 =    deeplabv3_resnet5zero(4)
    net4.load_state_dict(torch.load(model_file_list[3]))
    net4.to(device)
    net4.eval()

    # net5 =   deeplabv3_resnet5zero(4)
    net5.load_state_dict(torch.load(model_file_list[4]))
    net5.to(device)
    net5.eval()

    # net6 = modeling.__dict__['deeplabv3plus_resnet50'](num_classes=4)
    # net7 = modeling.__dict__['deeplabv3plus_resnet50'](num_classes=4)
    # net8 = modeling.__dict__['deeplabv3plus_resnet50'](num_classes=4)
    # net9 = modeling.__dict__['deeplabv3plus_resnet50'](num_classes=4)
    # net10 = modeling.__dict__['deeplabv3plus_resnet50'](num_classes=4)

    net6 =  deeplabv3_resnet5zero(4)
    net6.load_state_dict(torch.load(model_file_list1[0]))
    net6.to(device)
    net6.eval()

    net7 =    deeplabv3_resnet5zero(4)
    net7.load_state_dict(torch.load(model_file_list1[1]))
    net7.to(device)
    net7.eval()

    net8 =   deeplabv3_resnet5zero(4)
    net8.load_state_dict(torch.load(model_file_list1[2]))
    net8.to(device)
    net8.eval()

    net9 =    deeplabv3_resnet5zero(4)
    net9.load_state_dict(torch.load(model_file_list1[3]))
    net9.to(device)
    net9.eval()

    net10 =   deeplabv3_resnet5zero(4)
    net10.load_state_dict(torch.load(model_file_list1[4]))
    net10.to(device)
    net10.eval()

    net11 = modeling.__dict__['deeplabv3plus_resnet50'](num_classes=4)
    net12 = modeling.__dict__['deeplabv3plus_resnet50'](num_classes=4)
    net13 = modeling.__dict__['deeplabv3plus_resnet50'](num_classes=4)
    net14 = modeling.__dict__['deeplabv3plus_resnet50'](num_classes=4)
    net15 = modeling.__dict__['deeplabv3plus_resnet50'](num_classes=4)

    # net1 =  deeplabv3_resnet5zero(4)
    net11.load_state_dict(torch.load(model_file_list2[0]))
    net11.to(device)
    net11.eval()

    # net2 =    deeplabv3_resnet5zero(4)
    net12.load_state_dict(torch.load(model_file_list2[1]))
    net12.to(device)
    net12.eval()

    # net3 =   deeplabv3_resnet5zero(4)
    net13.load_state_dict(torch.load(model_file_list2[2]))
    net13.to(device)
    net13.eval()

    # net4 =    deeplabv3_resnet5zero(4)
    net14.load_state_dict(torch.load(model_file_list2[3]))
    net14.to(device)
    net14.eval()

    # net5 =   deeplabv3_resnet5zero(4)
    net15.load_state_dict(torch.load(model_file_list2[4]))
    net15.to(device)
    net15.eval()


    # print("self trained swin unet",msg)

    # snapshot_name = snapshot.split('/')[-1]


    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    from datasets.dataset_oct_goals import RandomGeneratorROI

    split = "testing"
    # split = "training"
    # split = "val"
    # "

    # db_test = args.Dataset(base_dir=args.volume_path, split=split, list_dir=args.list_dir,
    #                        transform=transforms.Compose(
    #                            [RandomGeneratorROI(output_size=[args.img_size, args.img_size])]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_test = OCTDataset(base_dir=args.volume_path, split="testing",
                                     transform=transforms.Compose(
                                         [RandomGeneratorROI(output_size=[args.img_size, args.img_size])]))

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1, pin_memory=False,
                           worker_init_fn=worker_init_fn, drop_last=False)


    print("lenlenlen", len(testloader))
    logging.info("{} test iterations per epoch".format(len(testloader)))


    metric_list = 0.0

    val_cup_dice = 0.0
    val_disc_dice = 0.0

    val_cup_pa = 0.0
    val_disc_pa = 0.0

    val_cup_iou = 0.0
    val_disc_iou = 0.0

    val_VCDR = 0.0
    val_MSE = 0.0

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(testloader):
            image_batch, label_batch = sampled_batch['oct_image'], sampled_batch['oct_mask']
            RNFL_batch, GCIPL_batch, Choriod_batch = sampled_batch['oct_RNFL'], \
                                                     sampled_batch['oct_GCIPL'], sampled_batch['oct_Choriod']
            case_name = sampled_batch['case_name']
            # print(case_name)

            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            B, C, H, W = image_batch.shape
            # print(image_batch.shape)

            # outputs = model(image_batch)
            # dice_loss = DiceLoss(4)
            # loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # print(case_name, loss_dice)

            # outputs = outputs['out']

            # outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)

            # outputs0 = net1(image_batch)
            # outputs1 = net2(image_batch)
            # outputs2 = net3(image_batch)
            # outputs3 = net4(image_batch)
            # outputs4 = net5(image_batch)
            #
            # outputs0 = torch.softmax(outputs0, dim=1)
            # outputs1 = torch.softmax(outputs1, dim=1)
            # outputs2 = torch.softmax(outputs2, dim=1)
            # outputs3 = torch.softmax(outputs3, dim=1)
            # outputs4 = torch.softmax(outputs4, dim=1)
            #
            # outputs12345 = (outputs0 + outputs1 + outputs2 + outputs3 + outputs4) / 5

            outputs11 = net11(image_batch)
            outputs12 = net12(image_batch)
            outputs13 = net13(image_batch)
            outputs14 = net14(image_batch)
            outputs15 = net15(image_batch)

            outputs11 = torch.softmax(outputs11, dim=1)
            outputs12 = torch.softmax(outputs12, dim=1)
            outputs13 = torch.softmax(outputs13, dim=1)
            outputs14 = torch.softmax(outputs14, dim=1)
            outputs15 = torch.softmax(outputs15, dim=1)

            outputs112345 = (outputs11 + outputs12 + outputs13 + outputs14 + outputs15) / 5

            outputs6 = net6(image_batch)
            outputs7 = net7(image_batch)
            outputs8 = net8(image_batch)
            outputs9 = net9(image_batch)
            outputs10 = net10(image_batch)

            outputs6 = outputs6['out']
            outputs7 = outputs7['out']
            outputs8 = outputs8['out']
            outputs9 = outputs9['out']
            outputs10 = outputs10['out']

            outputs6 = torch.softmax(outputs6, dim=1)
            outputs7 = torch.softmax(outputs7, dim=1)
            outputs8 = torch.softmax(outputs8, dim=1)
            outputs9 = torch.softmax(outputs9, dim=1)
            outputs10 = torch.softmax(outputs10, dim=1)

            outputs678910 = (outputs6 + outputs7 + outputs8 + outputs9 + outputs10) / 5



            outputs = (outputs112345 + outputs678910 + outputs112345 ) / 3

            # outputs = outputs12345


            outputs = torch.argmax(outputs, dim=1, keepdim=True)

            # print(outputs.shape,RNFL_batch.shape,GCIPL_batch.shape,Choriod_batch.shape)

            outputs = torch.squeeze(outputs)
            RNFL_batch = torch.squeeze(RNFL_batch)
            GCIPL_batch = torch.squeeze(GCIPL_batch)
            Choriod_batch = torch.squeeze(Choriod_batch)



            outputs = outputs.cpu().detach().numpy()
            RNFL_batch = RNFL_batch.cpu().detach().numpy()
            GCIPL_batch = GCIPL_batch.cpu().detach().numpy()
            Choriod_batch = Choriod_batch.cpu().detach().numpy()

            # outputs = zoom(outputs, (550 / 512, 1100 / 1024), order=0)
            # outputs = cv2.resize(o)

            # print(outputs.shape, RNFL_batch.shape, GCIPL_batch.shape, Choriod_batch.shape)

            outputs3_RNFL = np.zeros_like(outputs)
            outputs3_GCIPL = np.zeros_like(outputs)
            outputs3_Choriod = np.zeros_like(outputs)

            save_RGC = np.zeros_like(outputs)

            outputs3_RNFL[outputs == 0] = 1
            outputs3_GCIPL[outputs == 1] = 1
            outputs3_Choriod[outputs == 2] = 1

            save_RGC[outputs == 0] = 0
            save_RGC[outputs == 1] = 80
            save_RGC[outputs == 2] = 160
            save_RGC[outputs == 3] = 255

            print(case_name,save_RGC.shape)
            save_RGC1 = np.zeros([800,1100])
            # print(save_RGC1.shape)

            save_RGC1[save_RGC1==0]=255

            save_RGC1[50:600,:] = save_RGC[0:550,:]

            save_RGC1 = Image.fromarray(save_RGC1.astype(np.uint8))
            # odc = Image.fromarray(odc.astype(np.uint8))

            save_path_RGC1  = './datasets/goal/testing/pred/' +str(case_name[0])+'.png'
            # print(save_path_od)
            save_RGC1.save(save_path_RGC1)

   #  val_cup_dice = 0.0
   #  val_disc_dice = 0.0
   #
   #  val_cup_pa = 0.0
   #  val_disc_pa = 0.0
   #
   #  val_cup_iou = 0.0
   #  val_disc_iou = 0.0
   #
   #  val_VCDR = 0.0
   #  val_MSE = 0.0
   #
   #
   #  print(111)
   # ###交叉验证
   #  with torch.no_grad():
   #      for i_batch, sampled_batch in enumerate(testloader):
   #          image_batch, label_batch = sampled_batch['oct_image'], sampled_batch['oct_mask']
   #          RNFL_batch, GCIPL_batch, Choriod_batch = sampled_batch['oct_RNFL'], \
   #                                                   sampled_batch['oct_GCIPL'], sampled_batch['oct_Choriod']
   #          case_name = sampled_batch['case_name']
   #          # print(case_name)
   #
   #          image_batch, label_batch = image_batch.to(device), label_batch.to(device)
   #          B, C, H, W = image_batch.shape
   #          # print(image_batch.shape)
   #
   #
   #          outputs0, side_60, side_70 = net1(image_batch)
   #          outputs1, side_61, side_71 = net2(image_batch)
   #          outputs2, side_62, side_72 = net3(image_batch)
   #          outputs3, side_63, side_73 = net4(image_batch)
   #          outputs4, side_64, side_74 = net5(image_batch)
   #
   #          outputs0 = torch.softmax(outputs0, dim=1)
   #          outputs1 = torch.softmax(outputs1, dim=1)
   #          outputs2 = torch.softmax(outputs2, dim=1)
   #          outputs3 = torch.softmax(outputs3, dim=1)
   #          outputs4 = torch.softmax(outputs4, dim=1)
   #
   #          outputs = (outputs0 + outputs1 + outputs2 + outputs3 + outputs4) / 5
   #
   #
   #
   #          outputs = torch.argmax(outputs, dim=1, keepdim=True)
   #          # print(outputs.shape,RNFL_batch.shape,GCIPL_batch.shape,Choriod_batch.shape)
   #
   #          outputs = torch.squeeze(outputs)
   #          RNFL_batch = torch.squeeze(RNFL_batch)
   #          GCIPL_batch = torch.squeeze(GCIPL_batch)
   #          Choriod_batch = torch.squeeze(Choriod_batch)
   #
   #          outputs = outputs.cpu().detach().numpy()
   #          RNFL_batch = RNFL_batch.cpu().detach().numpy()
   #          GCIPL_batch = GCIPL_batch.cpu().detach().numpy()
   #          Choriod_batch = Choriod_batch.cpu().detach().numpy()
   #
   #          outputs = zoom(outputs, (1100 / 448, 550 / 224), order=0)
   #
   #          # print(outputs.shape, RNFL_batch.shape, GCIPL_batch.shape, Choriod_batch.shape)
   #
   #          outputs3_RNFL = np.zeros_like(outputs)
   #          outputs3_GCIPL = np.zeros_like(outputs)
   #          outputs3_Choriod = np.zeros_like(outputs)
   #
   #          save_RGC = np.zeros_like(outputs)
   #
   #          outputs3_RNFL[outputs == 0] = 1
   #          outputs3_GCIPL[outputs == 1] = 1
   #          outputs3_Choriod[outputs == 2] = 1
   #
   #          save_RGC[outputs == 0] = 0
   #          save_RGC[outputs == 1] = 80
   #          save_RGC[outputs == 2] = 160
   #          save_RGC[outputs == 3] = 255
   #
   #
   #
   #
   #          # import scipy
   #          # from skimage import morphology
   #          #
   #          # for i in range(5):
   #          #     disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
   #          #     cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
   #          #     # fovea_mask = scipy.signal.medfilt2d(fovea_mask, 7)
   #          # disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
   #          # cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
   #          # # fovea_mask = morphology.binary_erosion(fovea_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
   #          # od = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
   #          # oc = get_largest_fillhole(cup_mask).astype(np.uint8)
   #
   #          # print(save_RGC.shape)
   #          save_RGC1 = np.zeros([800, 1100])
   #          # print(save_RGC1.shape)
   #
   #          save_RGC1[save_RGC1 == 0] = 255
   #          save_RGC1[50:600, :] = save_RGC
   #
   #          save_RGC1 = Image.fromarray(save_RGC1.astype(np.uint8))
   #          # odc = Image.fromarray(odc.astype(np.uint8))
   #
   #          save_path_RGC1 = './datasets/goal/testing/pred/' + str(case_name[0]) + '.png'
   #          # print(save_path_od)
   #          save_RGC1.save(save_path_RGC1)



    val_cup_dice /= len(db_test)
    val_disc_dice /= len(db_test)
    val_disc_pa /= len(testloader)
    val_cup_pa /= len(testloader)
    val_cup_iou /= len(testloader)
    val_disc_iou /= len(testloader)
    val_VCDR /= len(testloader)

    print('''\n==>val_disc_dice : {0}'''.format(val_disc_dice))
    print('''\n==>val_cup_dice : {0}'''.format(val_cup_dice))
    print('''\n==>val_disc_pa : {0}'''.format(val_disc_pa))
    print('''\n==>val_cup_pa : {0}'''.format(val_cup_pa))
    print('''\n==>val_disc_iou : {0}'''.format(val_disc_iou))
    print('''\n==>val_cup_iou : {0}'''.format(val_cup_iou))
    print('''\n==>val_VCDR : {0}'''.format(val_VCDR))

    print("finished!")
    # inference(args, net, test_save_path)
