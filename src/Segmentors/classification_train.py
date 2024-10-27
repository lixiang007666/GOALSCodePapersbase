import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from torch.utils import model_zoo
from model.senet import se_resnet50
from utils.utils_fenlei import progress_bar


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')


parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.ImageFolder('/home/asus/Datasets/GOALS_challenges_miccai2022/GOALS2022-Train/Train/classification',
                                            transform=transform_train
                                            )
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=12)

testset = torchvision.datasets.ImageFolder('/home/asus/Datasets/GOALS_challenges_miccai2022/GOALS2022-Train/Train/classification',
                                            transform=transform_test
                                            )
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=12)



# Model
print('==> Building model..')

net = se_resnet50(2)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
model_ft = net.to(device)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()


if device == 'cuda':
    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint_senet154'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint_senet154/ckpt.pth')
    model_ft.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

def train(epoch,total_epoch):
    print('\nEpoch: %d' % epoch)
    model_ft.train()
    exp_lr_scheduler.step()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model_ft(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    if epoch==total_epoch-1:
        torch.save(model_ft.state_dict(), './checkpoint_se_resnet50_new_6_class/ckpt_final_epoch.pth')

def test(epoch):
    global best_acc
    model_ft.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_ft(inputs)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1) #第一个是值，第二个是索引。
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        if not os.path.isdir('checkpoint_se_resnet50_new_6_class'):
            os.mkdir('checkpoint_se_resnet50_new_6_class')
        torch.save(model_ft.state_dict(), './checkpoint_se_resnet50_new_6_class/ckpt.pth')
        best_acc = acc

batch = 16
for epoch in range(start_epoch, start_epoch+20): #resnet152用的是200epoch
    print(epoch)
    train(epoch,start_epoch+20)
    test(epoch)
