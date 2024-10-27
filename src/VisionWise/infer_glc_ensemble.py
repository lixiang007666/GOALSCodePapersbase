import time
import os
from torch import optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from utils.loader import GLC, GLC_Test
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
from skimage import filters
import logging
import time
import copy
import imageio
import numpy as np
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet34,resnet50,resnext50_32x4d,inception_v3,densenet121
import csv
torch.cuda.empty_cache()
cudnn.benchmark = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 4
n_epoch = 100

# writer = SummaryWriter(os.path.join('./dataset/JHUData/train/', 'train_exp', model_name+loss_name+times+extra_description))
np.set_printoptions(linewidth=200)

def visualize(image,mask):
	"""PLot images in one row."""
	# fig,axes = plt.subplots(1,2)
	plt.imshow(image[0,:,:],cmap='gray')
	plt.axis('off')
	plt.tight_layout()
	plt.show()
	plt.close()

def set_logger(log_file, log_level=logging.INFO):
	logger = logging.getLogger()  # 不加名称设置root logger
	logger.setLevel(log_level)
	formatter = logging.Formatter(
		'%(asctime)s %(levelname)s %(filename)s:%(lineno)d: %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S')

	# 使用FileHandler输出到文件
	fh = logging.FileHandler(log_file)
	fh.setLevel(log_level)
	fh.setFormatter(formatter)

	# 使用StreamHandler输出到屏幕
	ch = logging.StreamHandler()
	ch.setLevel(log_level)
	ch.setFormatter(formatter)

	# 添加两个Handler
	logger.addHandler(ch)
	logger.addHandler(fh)


model_name = 'inception_v3'

def main():
	torch.cuda.empty_cache()
	set_logger('train.log', log_level=logging.INFO)
	logging.info("+++++++++Training Start Here++++++++")
	# net = U_Net(img_ch=1, num_classes=6)
	
	# net = resnet50(pretrained=True)
	# net.fc = nn.Linear(2048, 2)
	model_paths = ['D:/PyDL/OCTseg-GOALS/checkpoint/glc/resnet50_pseudo_s1_best.pth',
				   'D:/PyDL/OCTseg-GOALS/checkpoint/glc/inception_v3_pseudo_s1_best.pth',
				   'D:/PyDL/OCTseg-GOALS/checkpoint/glc/resnet50_pseudo_s2_best.pth',
				   'D:/PyDL/OCTseg-GOALS/checkpoint/glc/inception_v3_pseudo_s2_best.pth']

	# train_set = GLC('D:/GOALS2022-All', 'train')
	# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	# val_set = GLC('D:/GOALS2022-All', 'val')
	# val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
	test_set = GLC_Test('D:/GOALS2022-Validation/Validation', 'test')
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
	
	# for id in range(len(val_set)):
	# 	image, mask, glc, path = val_set[id] # get some sample
	# 	visualize(image=image, mask=mask)

	infer(test_loader,model_paths, 'D:/PyDL/OCTseg-GOALS/output_glc/ensemble/resnet50_s1_inceptionv4_s1_resnet50_s2_inceptionv4_s2.csv')
	logging.info("----------Infer Finished Here----------") 

def infer(test_loader, model_paths, csvname):
	all_prob = []
	for model_path in model_paths:
		net = torch.load(model_path)
		net.cuda()
		net.eval()
		total_prob = []
		total_names = []
		with torch.no_grad():
			for inputs, names in test_loader:
				t1 = time.time()
				X = inputs.cuda()
				output_0 = net(X)
				prob = torch.softmax(output_0,dim=1)[:,1]
				# print(cat_output.max())
				prob = prob.detach().cpu().numpy()
				for j in range(len(names)):
					name = names[j].split('/')[-1]
					# name = name.split('.')[0]
					total_names.append(name)
					total_prob.append(prob[j])
		all_prob.append(total_prob)

	all_prob = np.array(all_prob)
	final_prob = np.mean(all_prob,axis=0)
	with open(csvname,'w',newline='') as f:
		writer = csv.writer(f,delimiter=',')
		writer.writerow(['ImgName','GC_Pred'])
		for i in range(len(total_names)):
			writer.writerow([total_names[i],final_prob[i]])

if __name__ == '__main__':
	# start = time.time
	main()
	# end = time.time
	# print('use time:', (end - start) // 3600)
