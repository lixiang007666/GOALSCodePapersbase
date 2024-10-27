import numpy as np
import torch
import torchvision.transforms as transforms
import os
from model.senet import se_resnet50
import cv2
from torch.autograd import Variable
import csv



transform_infer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
to_test = {'test':'/home/asus/Datasets/GOALS_challenges_miccai2022/GOALS2022-Validation/GOALS2022-Validation/Image/'}

def Resize(image,H, W):
    image = cv2.resize(image, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
    return image

def main():
    label = 0
    sum = 0
    net = se_resnet50(2,pretrained=False).cuda()
    net.load_state_dict(torch.load('./checkpoint_se_resnet50_new_6_class/ckpt_final_epoch.pth'))
    net.eval()
    error = []
    f = open('./Classification_Results_real.csv', 'w')
    writer = csv.writer(f)
    row = ['ImgName','GC_Pred']
    writer.writerow(row)
    with torch.no_grad():
        for name, root in to_test.items():
            root1 = os.path.join(root)
            img_list = [os.path.splitext(f) for f in os.listdir(root1) if f.endswith('.png')]
            img_list = sorted(img_list)
            print(img_list)
            for idx, img_name in enumerate(img_list):
                print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                rgb_png_path = os.path.join(root, img_name[0] + '.png')
                rgb_jpg_path = os.path.join(root, img_name[0] + '.jpg')
                rgb_bmp_path = os.path.join(root, img_name[0] + '.bmp')
                if os.path.exists(rgb_png_path):
                    img = cv2.imread(rgb_png_path, cv2.IMREAD_COLOR)[:, :, ::-1]
                elif os.path.exists(rgb_jpg_path):
                    img = cv2.imread(rgb_jpg_path, cv2.IMREAD_COLOR)[:, :, ::-1]
                else:
                    img = cv2.imread(rgb_bmp_path, cv2.IMREAD_COLOR)[:, :, ::-1]

                w_,h_,_ = img.shape
                img_resize = Resize(img,224,224)
                img_var = Variable(transform_infer(img_resize).unsqueeze(0), volatile=True).cuda()
                outputs = net(img_var)
                score, predicted = outputs.max(1)  # 第一个是值，第二个是索引。
                print(img_name[0])
                print(int(predicted.cpu()))
                row = [img_name[0]+'.png',int(predicted.cpu())]
                writer.writerow(row)

    f.close()
main()
