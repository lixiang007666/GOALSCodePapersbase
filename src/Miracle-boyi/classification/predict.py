import csv
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet34

def data_write_csv(file_name, datas, result):#file_name为写入CSV文件的路径，datas为要写入数据列表
    with open(file_name, "w", encoding="gbk", newline="") as f:
        # 2. 基于文件对象构建 csv写入对象
        csv_writer = csv.writer(f)
        # 3. 构建列表头
        csv_writer.writerow(["ImgName", "GC_Pred"])
        # 4. 写入csv文件内容
        for (x,y) in zip(datas,result):
            a = []
            a.append(x)
            a.append(y)
            csv_writer.writerow(a)

            # 5. 关闭文件
        f.close()
        print("写入数据成功")



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    all_img = os.listdir("./GOALS2022-Validation/Image")
    rootdir = "./GOALS2022-Validation/Image/"
    datas_1 = []
    reslut_1 = []
    for i in range(len(all_img)):
        # img_path = "../tulip.jpg"
        img_path = rootdir + all_img[i]
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            class_indict = json.load(f)

        # create model
        model = resnet34(num_classes=2).to(device)

        # load model weights
        weights_path = "./resNet34.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))

        # prediction
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

            # 结果
            datas_1.append(img_path[-8:])
            # reslut_1.append(round(predict.numpy()[1], 2))
            reslut_1.append(predict.numpy()[1])

    data_write_csv("./Classification_Results.csv", datas_1, reslut_1)
        # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
        #                                              predict[predict_cla].numpy())
        # plt.title(print_res)
        # for i in range(len(predict)):
        #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
        #                                               predict[i].numpy()))
        # plt.show()


if __name__ == '__main__':
    main()
