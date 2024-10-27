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


model_name = 'resnet50'

def main():
	torch.cuda.empty_cache()
	set_logger('train.log', log_level=logging.INFO)
	logging.info("+++++++++Training Start Here++++++++")
	# net = U_Net(img_ch=1, num_classes=6)
	
	net = resnet50(pretrained=True)
	net.fc = nn.Linear(2048, 2)

	
	# net = inception_v3(pretrained=True)
	# net.fc = nn.Linear(2048,2)

	net = net.cuda()

	# load
	# net = torch.load('./checkpoint/exp/unet_best.pth')

	train_set = GLC('D:/GOALS2022-All', 'train')
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	val_set = GLC('D:/GOALS2022-All', 'val')
	val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
	test_set = GLC_Test('D:/GOALS2022-Validation/Validation', 'test')
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
	
	criterion1 = nn.CrossEntropyLoss().cuda()
	criterion2 = nn.CrossEntropyLoss().cuda()
	optimizer = optim.Adam(net.parameters(), lr=3e-4,weight_decay=1e-4)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True, cooldown=2, min_lr=1e-8)

	print('Train:',len(train_set),' Val:',len(val_set))
	# for id in range(len(val_set)):
	# 	image, mask, glc, path = val_set[id] # get some sample
	# 	visualize(image=image, mask=mask)

	# test(test_loader,net,criterion1,criterion2,draw=True,cal_mae=False)
	train(train_loader, net, criterion1, criterion2, scheduler, optimizer, 0, n_epoch, val_loader)
	# infer(test_loader,net)
	logging.info("----------Training Finished Here----------") 

def val(val_loader, net, criterion1, criterion2, epoch):
	net.eval()
	val_loss = []
	total_prob = None
	total_gt = None
	with torch.no_grad():
		for inputs, masks, glc, names in val_loader:
			t1 = time.time()
			X = inputs.cuda()
			y = masks.cuda()
			glc = glc.cuda()
			
			# caution
			output_0 = net(X)

			loss = criterion1(output_0, glc) + criterion2(output_0, glc)
			val_loss.append(loss.item())
			prob = torch.softmax(output_0,dim=1) 
			
			val_loss.append(loss.item())
			prob = prob.detach().cpu().numpy()[:,1]
			glc = glc.detach().cpu().numpy()
			
			if total_prob is not None:
				total_prob = np.append(total_prob,prob,axis=0)
				total_gt = np.append(total_gt,glc,axis=0)
			else:
				total_prob = prob
				total_gt = glc
	auc = roc_auc_score(total_gt,total_prob)   
			  
	logging.info(f'Val AUC: {auc}')
		
	return np.mean(val_loss)

def infer(test_loader, net):

	net = torch.load('D:/PyDL/OCTseg-GOALS/checkpoint/glc/resnet50_pseudo_s1_ep100.pth')
	csvname = 'D:/PyDL/OCTseg-GOALS/output_glc/s1/resnet50_pseudo_s1_ep100.csv'
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
	with open(csvname,'w',newline='') as f:
		writer = csv.writer(f,delimiter=',')
		writer.writerow(['ImgName','GC_Pred'])
		for i in range(len(total_names)):
			writer.writerow([total_names[i],total_prob[i]])
	





def train(train_loader, net, criterion1, criterion2, scheduler, optimizer, iters, num_epoches,val_loader):
	net.train()
	train_loss = []
	best_loss = 100
	
	total_prob = None
	total_gt = None

	save_freq = 10

	for epoch in range(1, num_epoches+1):
		st = time.time()
		total_dcs = None
		for inputs, masks, glc, names in train_loader:
			X = inputs.cuda()
			y = masks.cuda()
			glc = glc.cuda()
			# X = X * y[:,1:2,:,:]

			optimizer.zero_grad()

			# caution
			output_0 = net(X)
			
			# convert to probability 0~1, channel direction

			loss = criterion1(output_0, glc) + criterion2(output_0, glc)

			prob = torch.softmax(output_0,dim=1) 
			
			loss.backward()
			optimizer.step()
			iters += batch_size
			train_loss.append(loss.item())
			
			prob = prob.detach().cpu().numpy()[:,1]
			glc = glc.detach().cpu().numpy()
			
			if total_prob is not None:
				total_prob = np.append(total_prob,prob,axis=0)
				total_gt = np.append(total_gt,glc,axis=0)
			else:
				total_prob = prob
				total_gt = glc
				
		
		#logging.debug(f'{total_dcs.shape}')
		auc = roc_auc_score(total_gt,total_prob)
		logging.info(f'Train Ep:{epoch}, AUC:{auc} {optimizer.param_groups[0]["lr"]}')
		train_mean_loss = np.mean(train_loss)

		#net = torch.load("cp_201110/{}.pth".format('U_Net_dice_no50_train10best'))
		#net = net.cuda()
		#train_mean_loss = np.mean(np.zeros((1,1)))
		
		val_loss = val(val_loader, net, criterion1,criterion2,epoch)
		net.train()
		scheduler.step(train_mean_loss)
		logging.info(f'Epoch {epoch}/{num_epoches},train_mean_loss {train_mean_loss:.4},val_mean_loss {val_loss:.4} ')
		if val_loss < best_loss:
			best_loss = val_loss
			logging.info(f'=====>best the model- loss:{best_loss}, epoch:{epoch}')
			torch.save(net, './checkpoint/glc/' + model_name + '_pseudo_s2_best.pth')
		# torch.save(net,'./checkpoint/exp/{}.pth'.format(epoch + val_loss))
		if epoch % save_freq == 0:
			torch.save(net, './checkpoint/glc/' + model_name + '_pseudo_s2_ep' + str(epoch) + '.pth')
		if epoch == num_epoches:
			torch.save(net, './checkpoint/glc/' + model_name + '_pseudo_s2_final.pth')




if __name__ == '__main__':
	# start = time.time
	main()
	# end = time.time
	# print('use time:', (end - start) // 3600)
