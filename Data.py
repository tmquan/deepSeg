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
def extractDataVec(images, radius=0):
	print ""
	print "Extracting vector of images..."
	numImages = images.shape[0]
	print "Number of images: ", numImages
	print "Radius: ", radius
	
	ret = []
	# Iterate over numImages
	for k in range(numImages):
		img = images[k]
		# print img.shape
		pad = np.pad(img, pad_width=radius, mode='symmetric')
		# print pad.shape
		
		# Retrive the dimensions of padded image
		[dimy, dimx] = pad.shape
		# cv2.imshow("", pad)
		# cv2.waitKey(0)
		
		for y in range(0, dimy, 1):
			for x in range(0, dimx, 1):
				if x<radius or x>(dimx-radius-1) or \
				   y<radius or y>(dimy-radius-1):
						continue;
				else:
					# Extract the patch and label
					patchImage = pad[y-radius:y+radius+1, \
									 x-radius:x+radius+1]; 
					ret.append(patchImage)
	ret = np.array(ret)
	ret = ret[:,np.newaxis,:,:]
	# ret = np.reshape(ret, (ret.shape[0], 	\
							# 1, 		\
							# ret.shape[1], 	\
							# ret.shape[2]))
	print ret.shape
	return ret
def data():
	# train_volume_file = "data/train-volume.tif"
	# train_labels_file = "data/train-labels.tif"
	# test_volume_file  = "data/test-volume.tif"
	# train_volume_file = "data/train_128x128x128.tif"
	train_volume_file = "data/l5_128x128x128.tif"
	train_labels_file = "data/segm_128x128x128.tif"
	test_volume_file  = "data/l5_128x128x128.tif"
	
	train_image = img2arr(train_volume_file)
	train_label = img2arr(train_labels_file)
	test_image  = img2arr(test_volume_file)
	
	# Extract vector of training data
	train_vec = extractDataVec(train_image, radius=16)
	np.save("X_train.npy", train_vec)
	del(train_vec)
	
	label_vec = extractDataVec(train_label, radius=0)
	np.save("y_train.npy", label_vec)
	del(label_vec)
	
	test_vec = extractDataVec(test_image, radius=16)
	np.save("X_test.npy",  test_vec)
	del(test_vec)
	# Save the data
	
	
	
	
	# train_volume_file = "data/train_128x128x128.tif"
	# train_labels_file = "data/segm_128x128x128.tif"
	
	# train_image = img2arr(train_volume_file)
	# train_label = img2arr(train_labels_file)
	
	# np.save("X_train.npy", train_image)
	# np.save("y_train.npy", train_label)
if __name__ == '__main__':
    data()
