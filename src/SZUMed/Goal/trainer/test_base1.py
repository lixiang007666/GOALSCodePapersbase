import os
import argparse
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from torch.autograd import Variable
from tqdm import tqdm
from data.ImageSet import ImageSet
import torch.backends.cudnn as cudnn
from efficientnet_pytorch import EfficientNet


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='GOAL')
parser.add_argument('--model', type=str, default='EfficientNet_b5')
parser.add_argument('--data_root', type=str,
                    default='/home/ubuntu/zhaobenjian/PyCharmProjects/data/GOAL/')
parser.add_argument('--project_root', type=str,
                    default='/home/ubuntu/zhaobenjian/PyCharmProjects/LocalHost/Goal_copy/')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
transform_val = transforms.Compose([
    transforms.Resize((550, 400)),
    transforms.ToTensor(),
])
testset = ImageSet(args.data_root, 'test', transform_val)
testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False)
cudnn.benchmark = True

print("Start testing ...")
with torch.no_grad():
    net = EfficientNet.from_name('efficientnet-b5')
    fc_num = net._fc.in_features
    net._fc = torch.nn.Linear(in_features=fc_num, out_features=2, bias=True)
    net = net.to(device)
    net.load_state_dict(torch.load(save_dir + 'models/best_weights-200.pth'))

    files_list = []
    ab_list_sm = []
    for batch_idx, (img, _, name) in enumerate(tqdm(testloader)):
        img = Variable(img.to(device))
        prob = net(img)
        prob_softmax = F.softmax(prob, dim=1)
        _, pred = torch.max(prob, 1)
        for idx in range(img.size(0)):
            files_list.append(name[idx])
            ab_list_sm.append(prob_softmax[idx, 1].item())

    cache = {'ImgName': files_list, 'GC_Pred': ab_list_sm}
    df = pd.DataFrame(cache)
    save_path_data = save_dir + 'Classification_Results.csv'
    df.to_csv(save_path_data, index=False)
    print("Ending")

