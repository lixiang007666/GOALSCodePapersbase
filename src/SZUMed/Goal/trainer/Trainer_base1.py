import os
import time
import argparse
import logging
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils.trainer_utils import dt
from utils.ConfusionMatrix import ConfusionMatrix
from utils.trainer_utils import train_log
from data.ImageSet import ImageSet
import torch.backends.cudnn as cudnn
from efficientnet_pytorch import EfficientNet


class Trainer:
    def __init__(self, model, optim, criterion, trainloader, validloader, epoch_num, save_dir, device, loss_write_dir):
        self.model = model
        self.optim = optim
        self.criterion_CE = criterion
        self.trainloader = trainloader
        self.validloader = validloader
        self.epoch_num = epoch_num
        self.save_dir = save_dir
        self.device = device
        self.writer = SummaryWriter(loss_write_dir)

    def train(self):
        logging.info('==' * 30)
        logging.info(dt())
        start_time = time.time()
        best_acc = 0.0
        for epoch in range(1, self.epoch_num+1):
            loss_total = 0.0
            count_total = 0
            for batch_idx, (img, label, name) in enumerate(tqdm(self.trainloader)):
                img = Variable(img.to(self.device))
                label = Variable(label.to(self.device))
                batch_size = label.shape[0]
                count_total += batch_size
                # train Classificator
                self.optim.zero_grad()
                logits = self.model(img)
                loss = self.criterion_CE(logits, label)
                loss.backward()
                self.optim.step()
                loss_total += loss.item()
                if (batch_idx + 1) % 10 == 0:
                    print("Epoch:%3d |Batch_idx:%3d |Loss:%.6f" % (epoch, batch_idx + 1, loss))
            self.writer.add_scalar('loss_train', loss_total / count_total, epoch)

            if epoch == 1:  # first batch
                path = self.save_dir + 'models/'
                if not os.path.exists(path):
                    os.makedirs(path)
                path = self.save_dir + 'predictions/'
                if not os.path.exists(path):
                    os.makedirs(path)

            acc = self.valid(epoch)

            if acc >= best_acc:  # save best model, acc
                best_acc = acc
                torch.save(self.model.state_dict(), self.save_dir + 'models/best_weights-%d.pth' % epoch)
                print('GOOD LUCK!!! Current best acc: %.4f, epoch: %d' % (acc, epoch))
            if epoch == self.epoch_num:  # save last models
                torch.save(self.model.state_dict(), self.save_dir + 'models/last_weights-%d.pth' % epoch)

            logging.info('Time elapsed: %4.2f' % ((time.time() - start_time) / 60))
            logging.info('--' * 30)

        self.writer.close()

    def valid(self, epoch):
        with torch.no_grad():
            files_list = []
            label_list = []
            pred_list = []
            ab_list_sm = []
            abnormal_list = []
            normal_list = []
            n_list_sm = []
            for batch_idx, (img, label, name) in enumerate(tqdm(self.validloader)):
                img = Variable(img.to(self.device))
                label = Variable(label.to(self.device))
                # validate Classificator
                logits = self.model(img)
                prob_softmax = F.softmax(logits, dim=1)
                _, pred = torch.max(logits, 1)
                for idx in range(label.size(0)):
                    files_list.append(name[idx])
                    abnormal_list.append(logits[idx, 1].item())  # logits output
                    ab_list_sm.append(prob_softmax[idx, 1].item())
                    normal_list.append(logits[idx, 0].item())
                    n_list_sm.append(prob_softmax[idx, 0].item())
                    label_list.append(label[idx].item())  # label
                    pred_list.append(pred[idx].item())  # prediction

            # metric
            metric_obj = ConfusionMatrix(num_classes=2, classes=['Normal', 'Abnormal'])
            metric_obj.update(pred_list, label_list)
            accuracy, precision, recall, specificity, F1, kappa = metric_obj.summary()
            auc = roc_auc_score(label_list, abnormal_list)

            print("Epoch:%3d |AUC:%.4f" % (epoch, auc))
            print(
                "Accuracy:%.4f |Precision:%.4f |Recall:%.4f |Specificity:%.4f |F1_score:%.4f" % (
                    accuracy, precision[1], recall[1], specificity[1], F1[1]))

            # log info
            logging.info("Epoch:%3d |AUC:%.4f" % (epoch, auc))
            logging.info(
                "Accuracy:%.4f |Precision:%.4f |Recall:%.4f |Specificity:%.4f |F1_score:%.4f" % (
                accuracy, precision[1], recall[1], specificity[1], F1[1]))

            if epoch == self.epoch_num:  # last epoch
                print(
                    "AUC:%.4f |Accuracy:%.4f |Precision:%.4f |Recall:%.4f |Specificity:%.4f |F1_score:%.4f" % (
                    auc, accuracy, precision[1], recall[1], specificity[1], F1[1]))

        save_output = {'file_name': files_list,
                       'logits': abnormal_list,
                       'prob_softmax': ab_list_sm,
                       'prediction': pred_list,
                       'label': label_list}
        df = pd.DataFrame(save_output)
        save_path_data = self.save_dir + 'predictions/output-%d.csv' % epoch
        df.to_csv(save_path_data, index=True)
        return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch_num', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dataset', type=str, default='GOAL')
    parser.add_argument('--model', type=str, default='EfficientNet_b5')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_root', type=str,
                        default='/home/ubuntu/zhaobenjian/PyCharmProjects/data/GOAL/')
    parser.add_argument('--project_root', type=str,
                        default='/home/ubuntu/zhaobenjian/PyCharmProjects/LocalHost/Goal_copy/')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # device_ids = [0, 1]
    device = 'cuda'

    if torch.cuda.is_available():
        print('We can use', torch.cuda.device_count(), 'GPUs to train the network!')
    else:
        print('We cannot use GPUs to train the network!')
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    print("Cuda Version: ", torch.version.cuda)
    print('================================================================================')

    save_dir = args.project_root + 'save/' + args.model + '/'
    loss_write__dir = args.project_root + 'runs/' + args.model + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(loss_write__dir):
        os.makedirs(loss_write__dir)
    logfile = "%s/trainlog.txt" % save_dir
    train_log(logfile)

    transform_train = transforms.Compose([
        transforms.Resize((550, 400)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((550, 400)),
        transforms.ToTensor(),
    ])
    trainset = ImageSet(args.data_root, 'train', transform_train)
    validset = ImageSet(args.data_root, 'valid', transform_val)
    trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    validloader = DataLoader(dataset=validset, batch_size=1, shuffle=False)

    cudnn.benchmark = True

    print('====>Construct EfficientNet..')
    net = EfficientNet.from_name('efficientnet-b5')
    state_dict = torch.load('../pytorch/efficientnet-b5.pth')
    net.load_state_dict(state_dict)

    fc_num = net._fc.in_features
    net._fc = torch.nn.Linear(in_features=fc_num, out_features=args.num_classes, bias=True)

    # net = nn.DataParallel(net, device_ids=device_ids)
    net = net.to(device)

    print('====>Set Optimizer...')
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    print('====>Set Criterion...')
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    # ================================================================================
    print('====>Start...')
    trainer = Trainer(
        model=net,
        optim=optim,
        criterion=criterion,
        trainloader=trainloader,
        validloader=validloader,
        epoch_num=args.epoch_num,
        save_dir=save_dir,
        device=device,
        loss_write_dir=loss_write__dir,
    )
    trainer.train()
