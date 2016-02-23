# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 10:12:52 2016

@author: tmquan
"""


from Utility import *
from Symbol import *
import random


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
def augment_image(img, lbl, shiftedDistance=0, rotatedAngle=0, flip=1, constrast=1):
	# First, transpose image to get normal numpy array
	# img = np.transpose(np.squeeze(img), (1, 2, 0))
	img = np.transpose((img), (1, 2, 0))
	lbl = np.transpose((lbl), (1, 2, 0))
	# print img.shape
	
	# Declare random option
	# random_translate = random.randint(-shiftedDistance, shiftedDistance)
	# random_rotatedeg = random.choice(range(-rotatedAngle, rotatedAngle))
	# random_flip      = random.choice([1, 2, 3, 4])
	
	#Random rotate images around center point which is randomly shifted
	if rotatedAngle !=0:
		random_rotatedeg = random.choice(range(-rotatedAngle, rotatedAngle))
		dimy, dimx, _ = img.shape
		rot_mat = cv2.getRotationMatrix2D(                          \
			(dimx/2+randint(-shiftedDistance, shiftedDistance), 	\
			 dimy/2+randint(-shiftedDistance, shiftedDistance)),    \
			random_rotatedeg, 1.0)
		rotated = np.zeros(img.shape)
		for k in range(img.shape[2]):
			rotated1[:,:,k] = cv2.warpAffine(img[:,:,k], rot_mat, (dimx, dimy))
			rotated2[:,:,k] = cv2.warpAffine(lbl[:,:,k], rot_mat, (dimx, dimy))
		img = rotated1
		lbl = rotated2
	img = img.astype(np.float32)
	lbl = lbl.astype(np.float32)
	if flip:
		random_flip      = random.choice([1, 2, 3, 4])
		flipped1 = np.zeros(img.shape)
		flipped2 = np.zeros(lbl.shape)
		for k in range(img.shape[2]):
			if random_flip==1:
				flipped1[:,:,k] = cv2.flip(img[:,:,k], -1)
				flipped2[:,:,k] = cv2.flip(lbl[:,:,k], -1)
			elif random_flip==2:
				flipped1[:,:,k] = cv2.flip(img[:,:,k], 0)
				flipped2[:,:,k] = cv2.flip(lbl[:,:,k], 0)
			elif random_flip==3:
				flipped1[:,:,k] = cv2.flip(img[:,:,k], 1)
				flipped2[:,:,k] = cv2.flip(lbl[:,:,k], 1)
			elif random_flip==4:
				flipped1[:,:,k] = img[:,:,k]
				flipped2[:,:,k] = lbl[:,:,k]
	else:
		flipped1 = img
		flipped2 = lbl 
	img = flipped1
	lbl = flipped2
	img = img.astype(np.float32)
	lbl = lbl.astype(np.float32)
	
	# dd = np.random.randint(3, 11) 
	# sS = np.random.randint(2, 50) 
	# sC = np.random.randint(2, 50)
	# if constrast:
		# img = denoise_tv_chambolle(img, weight=random.uniform(0.001, 0.1))
		# for k in range(img.shape[2]):
			# print img.dtype
			# print img.shape
			# img[:,:,k] = cv2.bilateralFilter(img[:,:,k], 
				# d=dd, 
				# sigmaSpace=sS, 
				# sigmaColor=sC
				# ) 
				
	img = np.transpose(img, (2, 0, 1))
	lbl = np.transpose(lbl, (2, 0, 1))
	return img, lbl 
def augment_data(X, y):
	progbar = Progbar(X.shape[0])
	for k in range(X.shape[0]):
		img  = X[k]
		lbl  = y[k]
		# print img.shape
		img, lbl  = augment_image(img, lbl)
		X[k] = img
		y[k] = lbl 
		progbar.add(1)
	return X, y 
		
def get_model():
	devs = [mx.gpu(0)]
	network = symbol_deconv()
	# arg_shape, output_shape, aux_shape = network.infer_shape(data=(1,30,256,256))
	# print "Shape", arg_shape, aux_shape, output_shape
	model = mx.model.FeedForward(ctx=devs,
		symbol          = network,
		num_epoch       = 1,
		learning_rate   = 0.001,
		wd              = 0.00001,
		initializer     = mx.init.Xavier(rnd_type="gaussian", 
							factor_type="in", 
							magnitude=1.0),
		momentum        = 0.9)	
	return model
    
def train():
	X = np.load('X_train.npy')
	y = np.load('y_train.npy')
	
	
	X = X.astype('float32')
	y = y.astype('float32')
	
	X = np.reshape(X, (30, 1, 512, 512))
	y = np.reshape(y, (30, 1, 512, 512))
	print np.max(y)
	y = y/np.max(y)
	print "X shape", X.shape
	print "X dtype", X.dtype
	print "Y shape", y.shape
	print "Y dtype", y.dtype
	
	
	nb_iter = 101
	epochs_per_iter = 1 
	batch_size = 10
	
	model_recon = get_model()
	# dot 		= mx.viz.plot_network(symbol_deconv())
	# print dot
	
	nb_folds = 3
	kfolds = KFold(len(y), nb_folds)
	for i in range(nb_iter):
		print('-'*50)
		print('Iteration {0}/{1}'.format(i + 1, nb_iter))  
		print('-'*50) 
		
		seed = i #np.random.randint(1, 10e6)
		f = 0
		for train, valid in kfolds:
			print('='*50)
			print('Fold', f+1)
			f += 1
			
			# Extract train, validation set
			X_train = X[train]
			X_valid = X[valid]
			y_train = y[train]
			y_valid = y[valid]
			
			print('Augmenting data for training...')
			X_train, y_train = augment_data(X_train, y_train) # Data augmentation for training 
			# print "X_train", X_train.shape
			# print "y_train", y_train.shape
			
			# print y_train
			# Convert to mxnet type
			X_train    		 = mx.nd.array(X_train)
			X_valid    		 = mx.nd.array(X_valid)
			y_train    		 = mx.nd.array(y_train)
			y_valid    		 = mx.nd.array(y_valid)
			
			
			# prepare data
			data_train = mx.io.NDArrayIter(X_train, y_train,
										   batch_size=batch_size, 
										   shuffle=True, 
										   last_batch_handle='roll_over'
										   )
			data_valid = mx.io.NDArrayIter(X_valid, y_valid,
										   batch_size=batch_size, 
										   shuffle=True, 
										   last_batch_handle='roll_over'
										   )
			
			# network = model_recon.symbol()
			# data_shape = (2,60,256,256)
			# arg_shape, output_shape, aux_shape = network.infer_shape(data=data_shape)
			# print "Shape", arg_shape, aux_shape, output_shape
	
			model_recon.fit(X = data_train, 
							eval_data = data_valid,
							eval_metric = mx.metric.RMSE()
							)
			if i%100==0:
				model_recon.save('models/model_recon', i)
if __name__ == '__main__':
	train()