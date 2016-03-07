# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 21:03:22 2016

@author: tmquan
"""

from Utility import *



def conv_factory(data, num_filter, kernel, stride, pad, act_type = 'relu', conv_type = 0):
    if conv_type == 0:
        conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
        bn = mx.symbol.BatchNorm(data=conv)
        act = mx.symbol.Activation(data = bn, act_type=act_type)
        return act
    elif conv_type == 1:
        conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
        bn = mx.symbol.BatchNorm(data=conv)
        return bn



def residual_factory(data, num_filter, dim_match):
    if dim_match == True: # if dimension match
        identity_data = data
        conv1 = conv_factory(data=data, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), act_type='relu', conv_type=0)
        
        conv2 = conv_factory(data=conv1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), conv_type=1)
        new_data = identity_data + conv2
        act = mx.symbol.Activation(data=new_data, act_type='relu')
        return act
    else:        
        conv1 = conv_factory(data=data, num_filter=num_filter, kernel=(3,3), stride=(2,2), pad=(1,1), act_type='relu', conv_type=0)
        conv2 = conv_factory(data=conv1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), conv_type=1)

        # adopt project method in the paper when dimension increased
        project_data = conv_factory(data=data, num_filter=num_filter, kernel=(1,1), stride=(2,2), pad=(0,0), conv_type=1)
        new_data = project_data + conv2
        act = mx.symbol.Activation(data=new_data, act_type='relu')
        return act

def residual_net(data, n):
    #fisrt 2n layers
    for i in range(n):
        data = residual_factory(data=data, num_filter=16, dim_match=True)
    
    #second 2n layers
    for i in range(n):
        if i==0:
            data = residual_factory(data=data, num_filter=32, dim_match=False)
        else:
            data = residual_factory(data=data, num_filter=32, dim_match=True)
    
    #third 2n layers
    for i in range(n):
        if i==0:
            data = residual_factory(data=data, num_filter=64, dim_match=False)
        else:
            data = residual_factory(data=data, num_filter=64, dim_match=True)
    return data
    
def symbol_resnet(numclass=1, workspace_default=1024):
	data = mx.symbol.Variable(name="data")
	data = data/255
	conv = conv_factory(
		data=data, 
		num_filter=16, 
		kernel=(3,3), 
		stride=(1,1), 
		pad=(1,1), 
		act_type='relu', 
		conv_type=0)
	n = 3 # set n = 3 means get a model with 3*6+2=20 layers, set n = 9 means 9*6+2=56 layers
	resnet 	= residual_net(conv, n) #	
	pool 	= mx.symbol.Pooling(data=resnet, kernel=(7,7), pool_type='avg')
	flatten = mx.symbol.Flatten(data=pool, name='flatten')
	fc = mx.symbol.FullyConnected(data=flatten, num_hidden=numclass,  name='fc1')
	softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
	return softmax

def symbol_deconv(numclass=1, workspace_default=1024):
	data = mx.symbol.Variable(name="data")
	# data = (data - 128) * (1.0/128)
	data = data/255
	
	conv = conv_factory(
		data=data, 
		num_filter=16, 
		kernel=(25,25), 
		stride=(1,1), 
		pad=(12,12), 
		act_type='relu', 
		conv_type=0)
	n = 5 # set n = 3 means get a model with 3*6+2=20 layers, set n = 9 means 9*6+2=56 layers
	rnet = residual_net(conv, n) # 
	# pool = mx.symbol.Pooling(data=resnet, kernel=(7,7), pool_type='avg')
	conv = conv_factory(
		data=rnet, 
		num_filter=64, 
		kernel=(1,1), 
		stride=(1,1), 
		pad=(0,0), 
		act_type='relu', 
		conv_type=0)
	# drop = mx.symbol.Dropout(data=conv, p=0.5)
	scale = 2
	pred1 = mx.symbol.Deconvolution(data=conv, 
		kernel=(2*scale, 2*scale), 
		stride=(scale, scale), 
		pad=(scale/2, scale/2), 
		num_filter=64, 
		no_bias=True, 
		workspace=workspace_default, 
		name='deconv_pred1')
	# drop = mx.symbol.Dropout(data=pred1, p=0.5)
	conv0 = conv_factory(
		data=pred1, 
		num_filter=64, 
		kernel=(1,1), 
		stride=(1,1), 
		pad=(0,0), 
		act_type='relu', 
		conv_type=0)
	pred2 = mx.symbol.Deconvolution(data=conv0, 
		kernel=(2*scale, 2*scale), 
		stride=(scale, scale), 
		pad=(scale/2, scale/2), 
		num_filter=1, 
		no_bias=True, 
		workspace=workspace_default, 
		name='deconv_pred2')
	# conv0 = conv_factory(
		# data=pred2, 
		# num_filter=numclass, 
		# kernel=(3,3), 
		# stride=(1,1), 
		# pad=(1,1), 
		# act_type='relu', 
		# conv_type=0)
	# flatten = mx.symbol.Flatten(conv)
	
	# fc0 = mx.symbol.FullyConnected(data=flatten, num_hidden=128,  name='fc0')
	# fc1 = mx.symbol.FullyConnected(data=fc0, num_hidden=16384,  name='fc1')
	# fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes,  name='fc1')
	# return mx.symbol.LogisticRegressionOutput(data=fc1, name='softmax')
	
	# softmax = mx.symbol.SoftmaxOutput(data=conv, multi_output=True, name="softmax")
	softmax = mx.symbol.LogisticRegressionOutput(data=pred2, name="softmax")
	
	return softmax
	
	# data = mx.symbol.Variable(name="data")
	# conv1 = conv_factory(data=data, 
		# num_filter=256, 
		# kernel=(3,3), 
		# stride=(1,1), 
		# pad=(1,1), 
		# act_type='relu', 
		# conv_type=0)
		
	# pool1 = mx.symbol.Pooling(data=conv1, 
							 # kernel=(2,2), 
							 # stride=(2,2), 
							 # pool_type='max')

	# conv2 = conv_factory(data=pool1, 
		# num_filter=256, 
		# kernel=(3,3), 
		# stride=(1,1), 
		# pad=(1,1), 
		# act_type='relu', 
		# conv_type=0)
		
	# pool2 = mx.symbol.Pooling(data=conv2, 
							 # kernel=(2,2), 
							 # stride=(2,2), 
							 # pool_type='max')

	# scale = 2
	# pred1 = mx.symbol.Deconvolution(data=pool2, 
		# kernel=(2*scale, 2*scale), 
		# stride=(scale, scale), 
		# pad=(scale/2, scale/2), 
		# num_filter=256, 
		# no_bias=True, 
		# workspace=workspace_default)
	# conv = conv_factory(data=pred1, 
		# num_filter=256, 
		# kernel=(1,1), 
		# stride=(1,1), 
		# pad=(0,0), 
		# act_type='relu', 
		# conv_type=0)
	# pred2 = mx.symbol.Deconvolution(data=conv, 
		# kernel=(2*scale, 2*scale), 
		# stride=(scale, scale), 
		# pad=(scale/2, scale/2), 
		# num_filter=256, 
		# no_bias=True, 
		# workspace=workspace_default)
	# conv = conv_factory(data=pred2, 
		# num_filter=numclass, 
		# kernel=(1,1), 
		# stride=(1,1), 
		# pad=(0,0), 
		# act_type='relu', 
		# conv_type=0)
	softmax = mx.symbol.SoftmaxOutput(data=pred1, name="softmax")
	# softmax = mx.symbol.LogisticRegressionOutput(data=conv, name="softmax")
	
	# return softmax

if __name__ == '__main__':
    # Draw the net
    network = symbol_deconv()
    dot = mx.viz.plot_network(network) 