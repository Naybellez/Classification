# 120923
# file for holding functions to keep notebooks clean

# Imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
from sklearn.model_selection import train_test_split

def import_imagedata(): # import image data from dir
	images = []
	labels = []

	file_path = r'/its/home/nn268/optics/images/'

	for file in os.listdir(file_path):
		if file[0:4] == 'IDSW':
			j = file_path+file
			i=int(file[5:7]) -1
			i = str(i)
			labels.append(i)
			images.append(j)
	label_arr =np.array(labels)
	image_arr = np.array(images)
	return image_arr, label_arr

def get_data():
	x, y = import_imagedata()
	random_seed = random.seed()
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_seed)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size =0.1, random_state=random_seed, shuffle=True)

	return x_train, y_train, x_val, y_val, x_test, y_test



def Unwrap(imgIn): #Amani unwrap fn

    def buildMap(Wd, Hd, R, Cx, Cy):
        ys=np.arange(0,int(Hd))
        xs=np.arange(0,int(Wd))

        rs=np.zeros((len(xs),len(ys)))
        rs=R*ys/Hd

        thetas=np.expand_dims(((xs-offset)/Wd)*2*np.pi,1)

        map_x=np.transpose(Cx+(rs)*np.sin(thetas)).astype(np.float32)
        map_y=np.transpose(Cy+(rs)*np.cos(thetas)).astype(np.float32)
        return map_x, map_y

    #UNWARP
    def Unwrap_(_img, xmap, ymap):
        output = cv2.remap(_img, xmap, ymap, cv2.INTER_LINEAR)
        return output


    img=cv2.resize(imgIn,None,fx=0.1,fy=0.1,interpolation=cv2.INTER_LINEAR)

    if img.shape[1] != img.shape[0]:
        cropBlock=int((int(img.shape[1])-int(img.shape[0]))/2)
        img=img[:,cropBlock:-cropBlock]

    #distance to the centre of the image
    offset=int(img.shape[0]/2)

    #IMAGE CENTER
    Cx = img.shape[0]/2
    Cy = img.shape[1]/2

    #RADIUS OUTER
    R =- Cx

    #DESTINATION IMAGE SIZE
    Wd = int(abs(2.0 * (R / 2)* np.pi))
    Hd = int(abs(R))

    #BUILD MAP
    xmap, ymap = buildMap(Wd, Hd, R, Cx, Cy)

    #UNWARP
    result = Unwrap_(img, xmap, ymap)

    return result
	
#preprocessing class. colour, scale, tensoring

class  ImageProcessor():
	def __init__(self, device):
		self.device=device

	#  colour functions
	def two_channels(self, g, r):
		new_im = [[],[]]
		new_im[0] = g
		new_im[1] = r
		new_im = np.array(new_im)
		new_im = np.transpose(new_im, (1,2,0))
		return new_im

	# tenor functions
	def tensoring(self, img):
		tense = torch.tensor(img, dtype=torch.float32)
		tense = F.normalize(tense)
		tense = tense.permute(2, 0, 1)
		return tense

	def to_tensor(self, img):
		im_chan = img.shape[2]
		imgY, imgX = img.shape[0], img.shape[1]
		tensor = self.tensoring(img)
		tensor = tensor.reshape(1, im_chan, imgY, imgX)
		tensor = tensor.to(self.device)
		return tensor

	#useful functions
	def colour_size_tense(self, img_path, col, size,  unwrapped=True):

		im = cv2.imread(img_path)
		if unwrapped:
			im = Unwrap(im)

		r = im[:,:,2]
		g = im[:,:,1]
		b = im[:,:,0]

		if col == 'nored':
			im = self.two_channels(b, g)
		elif col == 'noblue':
			im = self.two_channels(g, r)
		elif col == 'nogreen':
			im = self.two_channels(b, r)
		elif col == 'grey':
			im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		elif col =='colour' or col == 'color':
			pass
		
		im = cv2.resize(im,  (size[0], size[1]))
		im = self.to_tensor(im)
		#print(type(im))
		return im

	def view(self, img, scale:int):
		if type(img) == torch.Tensor:
			img = img.squeeze()
			img = img.permute(1,2,0)
			img=np.array(img.cpu())*scale
			plt.imshow(img)
			plt.axis(False)
			plt.show()
			return img


def label_oh_tf(lab, device):	# one hot encode label data
	num_classes = 11
	one_hot = np.zeros(num_classes)
	lab = int(lab)
	one_hot[lab] = 1
	label = torch.tensor(one_hot)
	label = label.to(torch.float32)
	label = label.to(device) #
	return label


def loop(model, X, Y, epoch, loss_fn, device, col_dict, optimizer =None, train =True):	# Train and Val loops. Default is train
	model = model
	total_samples = len(X)
	if train:
		model.train()
		#lr_ls = []
	else:
		model.eval()

	predict_list = []
	total_count = 0
	num_correct = 0
	current_loss = 0

	colour = col_dict['colour']
	size = col_dict['size']

	for idx, img in enumerate(X):
		#tense = tensoring(img).to(device)
		prepro = ImageProcessor(device, colour, size)
		tense = prepro.colour_size_tense(img)

		prediction = model.forward(tense)
		label = label_oh_tf(Y[idx], device)

		#if train:
		#	lr_ls.append(optimizer.param_groups[0]['lr'])

		loss = loss_fn(prediction, label)
		predict_list.append(prediction.argmax())

		if prediction.argmax() == label.argmax():
			num_correct +=1
			#if train:
			#	print(f'\n ########################### HIT ###########################  -- {idx} / {total_samples} \n')

		total_count+=1
		current_loss += loss.item()
		if train:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
	#print(num_correct/len(X))
	if train:
		return current_loss, predict_list, num_correct, model, optimizer #, lr_ls
	else:
		return current_loss, predict_list, num_correct

def test_loop(model, X, Y, loss_fn, device, col_dict,title, WANDB=False, wandb=None):
	model = model.eval()
	predict_list = []
	total_count =0
	num_correct = 0
	correct = 0
	colour = col_dict['colour']
	size = col_dict['size']

	with torch.no_grad():
		for idx, img in enumerate(X):
			prepro = ImageProcessor(device, colour, size)
			tense = prepro.colour_size_tense(img)
			prediction = model.forward(tense)
			label = label_oh_tf(Y[idx], device)

			if prediction.argmax()==label.argmax():
				num_correct +=1
			total_count +=1
			correct +=(prediction.argmax()==label.argmax()).sum().item()

		accuracy = 100*(num_correct/total_count)
		if WANDB:
			wandb.log({'Test_accuracy':accuracy})
			X = list(X)
			torch.onnx.export(model, X, f'{title}_accuracy{accuracy}.onnx')
			wandb.save(f'{title}_{accuracy}.onnx')





# Helpful printing functions
def print_run_header(learning_rate, optim, loss_fn):
	print('\n')
	print('LR: ', learning_rate)
	print('optimiser ', optim)
	print('loss fn: ', loss_fn)

def print_run_type(run_type: str):
	print('                  ----------------------')
	print(f' \n                  {run_type}... \n')
	print('                  ----------------------')

def check_best_accuracy(v_accuracy_list, best_valaccuracy):
	if v_accuracy_list[-1] > best_valaccuracy:
		best_valaccuracy = v_accuracy_list[-1]
		best_optim = optimizer
		best_lossfn = loss_fn
		best_lr = learning_rate
		best_epoch = epoch
	return best_valaccuracy, best_optim, best_lossfn, best_lr, best_epoch

def print_top_results(best_optim, best_lossfn, best_lr, best_valaccuracy, best_epoch):
	print('Top results from hyperparameter sweep:')
	print()
	print(best_optim, best_lossfn, best_lr, best_valaccuracy, best_epoch)

def set_optimizer(optim):
	optim_list=[]
	if optim =='Adam':
		optimizer1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
		optimizer2 = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
		optim_list.append(optimizer1)
		optim_list.append(optimizer2)
	elif optim == 'SGD':
		optimizer3 = torch.optim.SGD(model.parameters(), lr=learning_rate)
		optim_list.append(optimizer3)
	return optim_list



