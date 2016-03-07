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
# def augment_image(img, lbl, shiftedDistance=0, rotatedAngle=1, flip=1, constrast=1):
	# # First, transpose image to get normal numpy array
	# # img = np.transpose(np.squeeze(img), (1, 2, 0))
	# img = np.transpose((img), (1, 2, 0))
	# lbl = np.transpose((lbl), (1, 2, 0))
	# # print img.shape
	
	# # Declare random option
	# # random_translate = random.randint(-shiftedDistance, shiftedDistance)
	# # random_rotatedeg = random.choice(range(-rotatedAngle, rotatedAngle))
	# # random_flip      = random.choice([1, 2, 3, 4])
	
	# #Random rotate images around center point which is randomly shifted
	# if rotatedAngle !=0:
		# # random_rotatedeg = random.choice(range(-rotatedAngle, rotatedAngle))
		# random_rotatedeg = random.choice([0, 90, 180, 270])
		# dimy, dimx, _ = img.shape
		# rot_mat = cv2.getRotationMatrix2D(                          \
			# (dimx/2+randint(-shiftedDistance, shiftedDistance), 	\
			 # dimy/2+randint(-shiftedDistance, shiftedDistance)),    \
			# random_rotatedeg, 1.0)
		# rotated1 = np.zeros(img.shape)
		# rotated2 = np.zeros(img.shape)
		# for k in range(img.shape[2]):
			# rotated1[:,:,k] = cv2.warpAffine(img[:,:,k], rot_mat, (dimx, dimy))
			# rotated2[:,:,k] = cv2.warpAffine(lbl[:,:,k], rot_mat, (dimx, dimy))
		# img = rotated1
		# lbl = rotated2
	# img = img.astype(np.float32)
	# lbl = lbl.astype(np.float32)
	# if flip:
		# random_flip      = random.choice([1, 2, 3, 4])
		# flipped1 = np.zeros(img.shape)
		# flipped2 = np.zeros(lbl.shape)
		# for k in range(img.shape[2]):
			# if random_flip==1:
				# flipped1[:,:,k] = cv2.flip(img[:,:,k], -1)
				# flipped2[:,:,k] = cv2.flip(lbl[:,:,k], -1)
			# elif random_flip==2:
				# flipped1[:,:,k] = cv2.flip(img[:,:,k], 0)
				# flipped2[:,:,k] = cv2.flip(lbl[:,:,k], 0)
			# elif random_flip==3:
				# flipped1[:,:,k] = cv2.flip(img[:,:,k], 1)
				# flipped2[:,:,k] = cv2.flip(lbl[:,:,k], 1)
			# elif random_flip==4:
				# flipped1[:,:,k] = img[:,:,k]
				# flipped2[:,:,k] = lbl[:,:,k]
	# else:
		# flipped1 = img
		# flipped2 = lbl 
	# img = flipped1
	# lbl = flipped2
	# img = img.astype(np.float32)
	# lbl = lbl.astype(np.float32)
	
	# # dd = np.random.randint(3, 11) 
	# # sS = np.random.randint(2, 50) 
	# # sC = np.random.randint(2, 50)
	# # if constrast:
		# # img = denoise_tv_chambolle(img, weight=random.uniform(0.001, 0.1))
		# # for k in range(img.shape[2]):
			# # print img.dtype
			# # print img.shape
			# # img[:,:,k] = cv2.bilateralFilter(img[:,:,k], 
				# # d=dd, 
				# # sigmaSpace=sS, 
				# # sigmaColor=sC
				# # ) 
				
	# img = np.transpose(img, (2, 0, 1))
	# lbl = np.transpose(lbl, (2, 0, 1))
	# return img, lbl 
# def augment_data(X, y):
	# progbar = Progbar(X.shape[0])
	# for k in range(X.shape[0]):
		# img  = X[k]
		# lbl  = y[k]
		# # print img.shape
		# img, lbl  = augment_image(img, lbl)
		# X[k] = img
		# y[k] = lbl 
		# progbar.add(1)
	# return X, y 
	
def augment_image(img, shiftedDistance=0, rotatedAngle=1, flip=1, constrast=1):
	# First, transpose image to get normal numpy array
	# img = np.transpose(np.squeeze(img), (1, 2, 0))
	img = np.transpose((img), (1, 2, 0))
	# print img.shape
	
	# Declare random option
	# random_translate = random.randint(-shiftedDistance, shiftedDistance)
	# random_rotatedeg = random.choice(range(-rotatedAngle, rotatedAngle))
	# random_flip      = random.choice([1, 2, 3, 4])
	
	#Random rotate images around center point which is randomly shifted
	if rotatedAngle !=0:
		# random_rotatedeg = random.choice(range(-rotatedAngle, rotatedAngle))
		random_rotatedeg = random.choice([0, 90, 180, 270])
		dimy, dimx, _ = img.shape
		rot_mat = cv2.getRotationMatrix2D(                          \
			(dimx/2+randint(-shiftedDistance, shiftedDistance), 	\
			 dimy/2+randint(-shiftedDistance, shiftedDistance)),    \
			random_rotatedeg, 1.0)
		rotated1 = np.zeros(img.shape)
		for k in range(img.shape[2]):
			rotated1[:,:,k] = cv2.warpAffine(img[:,:,k], rot_mat, (dimx, dimy))
		img = rotated1
	img = img.astype(np.float32)
	if flip:
		random_flip      = random.choice([1, 2, 3, 4])
		flipped1 = np.zeros(img.shape)
		for k in range(img.shape[2]):
			if random_flip==1:
				flipped1[:,:,k] = cv2.flip(img[:,:,k], -1)
			elif random_flip==2:
				flipped1[:,:,k] = cv2.flip(img[:,:,k], 0)
			elif random_flip==3:
				flipped1[:,:,k] = cv2.flip(img[:,:,k], 1)
			elif random_flip==4:
				flipped1[:,:,k] = img[:,:,k]
	else:
		flipped1 = img
	img = flipped1
	img = img.astype(np.float32)
	
				
	img = np.transpose(img, (2, 0, 1))
	return img
	
def augment_data(X):
	progbar = Progbar(X.shape[0])
	for k in range(X.shape[0]):
		img  = X[k]
		# print img.shape
		img  = augment_image(img)
		X[k] = img
		progbar.add(1)
	return X
	
def get_model():
	# devs = [mx.gpu(2)]
	# devs = [mx.gpu(i) for i in range(3)]
	devs = [mx.gpu(0), mx.gpu(3)]
	network = symbol_resnet()
	# arg_shape, output_shape, aux_shape = network.infer_shape(data=(1,30,256,256))
	# print "Shape", arg_shape, aux_shape, output_shape
	model = mx.model.FeedForward(ctx=devs,
		symbol          = network,
		num_epoch       = 1,
		learning_rate   = 0.001,
		wd              = 0.00001,
		initializer     = mx.init.Xavier(rnd_type="gaussian", 
							factor_type="in", 
							magnitude=2.34),
		momentum        = 0.9)	
	return model
    
def train():
	X = np.load('X_train.npy')
	y = np.load('y_train.npy')
	y = y.flatten(order='C')
	
	X = X.astype('float32')
	y = y.astype('float32')
	
	# X = np.reshape(X, (30, 1, 512, 512))
	# y = np.reshape(y, (30, 1, 512, 512))
	
	# X = np.reshape(X, (128, 1, 128, 128))
	# y = np.reshape(y, (128, 1, 128, 128))
	
	
	
	print np.max(y)
	# X = X/255
	y  = y/255
	
	# y0 = y==0
	# # y0 = y0[None,:]
	# y1 = y>0
	# # y1 = y1[None,:]
	# y  = np.stack((y0, y1),axis=0)
	# y  = np.transpose(y)
	y  = y.astype(np.float32)
	
	print y.shape
	print "X shape", X.shape
	print "X dtype", X.dtype
	print "Y shape", y.shape
	print "Y dtype", y.dtype
	
	
	nb_iter = 101
	epochs_per_iter = 1 
	batch_size = 4096
	
	model_recon = get_model()
	# dot 		= mx.viz.plot_network(symbol_deconv())
	# print dot
	
	nb_folds = 4
	kfolds = KFold(len(y), nb_folds)
	for i in range(nb_iter):
		print('-'*50)
		print('Iteration {0}/{1}'.format(i + 1, nb_iter))  
		print('-'*50) 
		
		# seed = i #np.random.randint(1, 10e6)
		# Shuffle the data
		print('Shuffle data...')
		seed = np.random.randint(1, 10e6)
		np.random.seed(seed)
		np.random.shuffle(X)
		np.random.seed(seed)
		np.random.shuffle(y)
		
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
			
			# print('Augmenting data for training...')
			# X_train = augment_data(X_train) # Data augmentation for training 
			# X_valid = augment_data(X_valid) # Data augmentation for training 
			# print "X_train", X_train.shape
			# print "y_train", y_train.shape
			
			# y_train = np.reshape(y_train, (96, 128*128))
			# y_valid = np.reshape(y_valid, (32, 128*128))
			
			# print y_train
			# Convert to mxnet type
			X_train    		 = mx.nd.array(X_train)
			X_valid    		 = mx.nd.array(X_valid)
			y_train    		 = mx.nd.array(y_train)
			y_valid    		 = mx.nd.array(y_valid)
			

			# prepare data
			data_train = mx.io.NDArrayIter(X_train, y_train,
										   batch_size=batch_size, 
										   shuffle=False, 
										   last_batch_handle='pad'
										   )
			data_valid = mx.io.NDArrayIter(X_valid, y_valid,
										   batch_size=batch_size, 
										   shuffle=False, 
										   last_batch_handle='pad'
										   )
			
			# network = model_recon.symbol()
			# data_shape = (96,1,128,128)
			# arg_shape, output_shape, aux_shape = network.infer_shape(data=data_shape)
			# print "Shape", arg_shape, aux_shape, output_shape
			# def norm_stat(d):
				# return mx.nd.norm(d)/np.sqrt(d.size)
			# mon = mx.mon.Monitor(100, norm_stat)
			model_recon.fit(X = data_train, 
							# eval_metric = mx.metric.RMSE(),
							# eval_metric = mx.metric.Accuracy(),
							# eval_metric = mx.metric.CustomMetric(skimage.measure.compare_psnr),
							# eval_metric = mx.metric.MAE(),
							eval_data = data_valid,
							batch_end_callback = mx.callback.Speedometer(batch_size, batch_size)
							# monitor=mon 
							)
		if i%10==0:
			model_recon.save('models/model_recon', i)
if __name__ == '__main__':
	train()