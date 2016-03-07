from Utility import *

def deploy():
	X = np.load('X_test.npy')
	
	X = np.reshape(X, (30, 1, 512, 512))
	print "X.shape", X.shape
	X_deploy = X[24:25,:,:,:]
	print "X_deploy.shape", X_deploy.shape
	# Load model
	iter = 100 
	model_recon 	= mx.model.FeedForward.load('models/model_recon', iter, ctx=mx.gpu(0))
	# model_recon 	= mx.model.FeedForward.load('models/model_recon', iter)
	
	network = model_recon.symbol()
	arg_shape, output_shape, aux_shape = network.infer_shape(data=(1,1,512,512))
	print "Shape", arg_shape, aux_shape, output_shape
	
	# Perform prediction
	batch_size = 1
	print('Predicting on data...')
	pred_recon  = model_recon.predict(X_deploy, num_batch=None)
	
	pred_recon  = np.array(pred_recon)
	pred_recon  = np.reshape(pred_recon, (512, 512))
	plt.imshow((np.absolute(pred_recon)) , cmap = plt.get_cmap('gray'))
	plt.show()	
	
if __name__ == '__main__':
	deploy()
