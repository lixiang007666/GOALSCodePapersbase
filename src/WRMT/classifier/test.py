import numpy as np
import sys, random
import torch
from torchvision import models, transforms
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
# Paths for image directory and model
IMDIR='G:/MICCAI/GOALS2022-Validation/Image'
MODEL='G:/MICCAI/pytorch-image-classification-master/logs/model.pth'

# Load the model for testing
model = torch.load(MODEL)
model.eval()

# Class labels for prediction
# class_names=['0','atm card','cat','banana','bangle','battery','bottle','broom','bulb','calender','camera']
# class_names=['0','1']

# Retreive 9 random images from directory
# files=Path(IMDIR).resolve().glob('*.*')
# images=random.sample(list(files), 9)
# images =Path(IMDIR).resolve().glob('*.*')
# print(images)

# Configure plots
# fig = plt.figure(figsize=(9,9))
# rows,cols = 3,3

# Preprocessing transformations
preprocess=transforms.Compose([
        # transforms.Resize(size=256),
        # transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

name=[]
predict_result=[]
# Perform prediction and plot results
with torch.no_grad():
    for i in os.listdir(IMDIR):
        img_file = os.path.join(IMDIR,i)
        img=Image.open(img_file).convert('RGB')
        inputs=preprocess(img).unsqueeze(0).to(device)
        outputs = model(inputs)
        outputs = torch.softmax(outputs,1)
        outputs = outputs.cpu().numpy()
        name.append(i)
        predict_result.append(outputs[0][1])
dataframe = pd.DataFrame({'ImgName': name, 'GC_Pred': predict_result})
csv_name = 'G:/MICCAI/pytorch-image-classification-master/results/Classification_Results.csv'
dataframe.to_csv(csv_name, index=False,sep=',', header=None)
