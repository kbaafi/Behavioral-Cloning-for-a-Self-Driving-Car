#from model import *
from bc_preprocessing import *
from model import *

import numpy as np
import cv2
import pandas as pd
import pickle

from keras.models import model_from_json

data_path = 'CAR_data/data/'
img_path = 'IMG/'
csv_path = 'driving_log.csv'

def bc_train_nvidia():
	"""
	Trains a neural network model using a data generator
	"""
	img_rows,img_cols = 64,64
	input_shape = (img_rows,img_cols,3)

	# the model	
	model = bc_nvidia_model(input_shape = input_shape)

	
	img_dim = (img_rows,img_cols)

	# reading the drivelog	
	csv_data = pd.read_csv(data_path+csv_path,usecols=["center","left","right","steering"])

	threshold = 1
	batch_size = 240
	epochs = 6
	yvals = []

	for i in range(epochs):
		gen = generate_data_train(data_path,csv_data,img_dim,batch_size,threshold,yvals)
		
		model.fit_generator(gen, samples_per_epoch = 24000, nb_epoch = 1, verbose = 1)

		# thresholding against values close to 0 to balance the data
		threshold = 1/(i+1)
	
	# serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")
	with open("s_angles","wb") as y_file:
		pickle.dump(yvals,y_file)
	return

if __name__ == '__main__':
	bc_train_nvidia()
