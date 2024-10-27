import os
import pandas as pd
import shutil
img_path = r'G:\MICCAI\Train\Image'
lab_path = r'G:\MICCAI\Train\Train_GC_GT.xlsx'
save_path =r'G:\MICCAI\Train\train'
id = []
for i in os.listdir(img_path):
    id.append(i.split('.')[0])
# print(id)
df = pd.read_excel(lab_path)
label_id = df['GC_Label'].tolist()
for i in range(len(label_id)):
    img_file = os.path.join(img_path,id[i]+".png")
    if label_id[i] == 0:
        save_file = os.path.join(save_path,'0',id[i]+".png")
        shutil.copy(img_file,save_file)
    if label_id[i] == 1:
        save_file = os.path.join(save_path,'1',id[i]+".png")
        shutil.copy(img_file,save_file)
    print(label_id[i],'is done!')
