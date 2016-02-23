# Import all necessary packages
from Utility import *

def img2arr(imageFile):
	"""
	Read images tif files and store as numpy array
	"""
	img = skimage.io.imread(imageFile)
	print "File name: ", imageFile
	print "Shape    : ", img.shape
	return img
def data():
	train_volume_file = "data/train-volume.tif"
	train_labels_file = "data/train-labels.tif"
	test_volume_file  = "data/test-volume.tif"
	
	train_image = img2arr(train_volume_file)
	train_label = img2arr(train_labels_file)
	test_image  = img2arr(test_volume_file)
	
	np.save("X_train.npy", train_image)
	np.save("y_train.npy", train_label)
	np.save("X_test.npy",  test_image)
	
	
if __name__ == '__main__':
    data()
