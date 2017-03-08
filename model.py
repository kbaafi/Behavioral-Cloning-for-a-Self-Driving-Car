"""
	Implementations of learning models that are tested for suitability to 
	behavioral cloning for autonomous driving
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Lambda,Cropping2D
from keras.utils import np_utils
from keras.layers.core import Reshape
#from keras.optimizers import SGD, Adam, RMSprop

import numpy as np

def bc_nvidia_model(input_shape):
	"""
	Implements the model presented by NVIDIA in their paper here:
		https://arxiv.org/pdf/1604.07316.pdf
	Args:
		input_shape: input shape of image after preprocessing expected 
				value is 66x200. User may need to adjust this
				model to cater for different sizes or simply create
				a new model based on this
	Returns:
		model: constructed CNN
	"""
	model = Sequential()
	#print("input shape",input_shape)
	# Normalization Layer
	model.add(Lambda(lambda x: x/255 - 0.5,input_shape = input_shape))
	
	# Convolutional Layers	
	model.add(Convolution2D(24,5,5,subsample = (2,2), border_mode = 'valid', init = 'he_normal'))
	model.add(Activation('relu'))
	

	model.add(Convolution2D(36,5,5,subsample = (2,2), border_mode = 'valid', init = 'he_normal'))
	model.add(Activation('relu'))
	

	model.add(Convolution2D(48,5,5,subsample = (2,2), border_mode = 'valid', init = 'he_normal'))
	model.add(Activation('relu'))
	

	model.add(Convolution2D(64,3,3,subsample = (1,1), border_mode = 'valid', init = 'he_normal'))
	model.add(Activation('relu'))
	

	model.add(Convolution2D(64,3,3,subsample = (1,1), border_mode = 'valid', init = 'he_normal'))
	model.add(Activation('relu'))

	# Flatten	
	model.add(Flatten())
	
	# Fully Connected Layers
	model.add(Dense(1164, init = 'he_normal'))
	model.add(Activation('relu'))

	model.add(Dense(100, init = 'he_normal'))
	model.add(Activation('relu'))

	model.add(Dense(50, init = 'he_normal'))
	model.add(Activation('relu'))

	model.add(Dense(10, init = 'he_normal'))
	model.add(Activation('relu'))

	model.add(Dense(1, init = 'he_normal'))

	model.compile(optimizer = 'adadelta', loss = 'mse')
	
	return model
	
