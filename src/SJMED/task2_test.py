import numpy as np
import pandas as pd
import torch
import torchvision.transforms as trans
from models.task2_model import Model
from dataset_functions.task2_dataset import GOALS_sub2_dataset



# 预测阶段
best_model_path = './checkpoints/task2/train1.pth'
image_size = 256
test_root = '../datasets/Validation/Image'

model = Model()
para_state_dict = torch.load(best_model_path)
model.load_state_dict(para_state_dict)
model.eval()

img_test_transforms = trans.Compose([
    trans.ToTensor(),
    trans.Resize((image_size, image_size))
])

test_dataset = GOALS_sub2_dataset(dataset_root=test_root, 
                        img_transforms=img_test_transforms,
                        mode='test')
cache = []
for img, idx in test_dataset:
    img = img.unsqueeze(0) # 增加维度：(3,256,256)-->(1,3,256,256)
    # print(img.shape)
    logits = model(img) 
    # _, indice = logits.max(dim=1) # 找出行最大值，返回索引
    m = torch.nn.Softmax()
    pred = m(logits)
    print(pred)
    idx = '0' + str(idx) + '.png'
    cache.append([idx, pred.detach().numpy()[0][1]])

submission_result = pd.DataFrame(cache, columns=['ImgName', 'GC_Pred'])
submission_result[['ImgName', 'GC_Pred']].to_csv("./submission/task2/train1.csv", index=False)