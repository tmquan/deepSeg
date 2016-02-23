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
	batch_size = 30
	
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
										   shuffle=False, 
										   last_batch_handle='roll_over'
										   )
			data_valid = mx.io.NDArrayIter(X_valid, y_valid,
										   batch_size=batch_size, 
										   shuffle=False, 
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