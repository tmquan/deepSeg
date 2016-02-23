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
	
def symbol_deconv(numclass=1, workspace_default=2048):
	data = mx.symbol.Variable(name="data")
	conv = conv_factory(data=data, 
		num_filter=256, kernel=(1,1), stride=(1,1), pad=(0,0), act_type='relu', conv_type=0)
		
	pool = mx.symbol.Pooling(data=conv, 
							 kernel=(2,2), 
							 stride=(2,2), 
							 pool_type='max')
	# deconv = mx.symbol.Deconvolution(data=pool, 
									 # num_filter=1, 
									 # kernel = (2,2), 
									 # stride = (2,2), 
									 # pad	= (0,0),									 
									 # workspace=workspace_default, 
									 # name="deconv")
	scale = 2
	pred1 = mx.symbol.Deconvolution(data=pool, 
		kernel=(2*scale, 2*scale), 
		stride=(scale, scale), 
		pad=(scale/2, scale/2), 
		num_filter=numclass, 
		no_bias=True, 
		workspace=workspace_default, 
		name='deconv_pred1')
	# softmax = mx.symbol.SoftmaxOutput(data=pred1, name="softmax")
	softmax = mx.symbol.LogisticRegressionOutput(data=pred1, name="softmax")
	
	return softmax

if __name__ == '__main__':
    # Draw the net
    network = symbol_deconv()
    dot = mx.viz.plot_network(network) 