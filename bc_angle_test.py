import numpy as np
import cv2
from bc_augmentation import *
import math
import pandas as pd

def aug_angle_tr(steering_angle,x_translation_range,y_translation_range):
	x_translation = (x_translation_range*np.random.uniform())-(x_translation_range/2)
	y_translation = (y_translation_range*np.random.uniform())-(y_translation_range/2)
	
	new_steering_angle = steering_angle+((x_translation/x_translation_range)*2)*0.2
	return new_steering_angle

def aug_angle_flip(steering_angle):
	return -steering_angle

def get_augmented_angles(adata):
	
	angles = []
	for element in adata:
		angles.append(element)
		angles.append(element)
		tr_angle = aug_angle_tr(element,100,40)
		angles.append(tr_angle)
		fl_angle = -1*element
		angles.append(fl_angle)
	return angles

def preprocess_data(csvline):
	steering_angle = float(csvline['steering'][0])
	left_angle = steering_angle+0.2
	right_angle = steering_angle-0.2
	sdata = []
	sdata.append(steering_angle)
	sdata.append(left_angle)
	sdata.append(right_angle)

	angles = get_augmented_angles(sdata)
	return angles
def augment_steering(sangle):
	angles = [];
	angles.append(sangle)
	angles.append(sangle)
	tr_angle = aug_angle_tr(sangle,100,40)
	angles.append(tr_angle)
	fl_angle = -1*sangle
	angles.append(fl_angle)
	return angles

def process_data_2():
	data_path = 'CAR_data/data/'
	img_path = 'IMG/'
	csv_path = 'driving_log.csv'

	csv_data = pd.read_csv(data_path+csv_path,usecols=["center","left","right","steering"])

	threshold = 1
	batch_size = 600
	epochs = 10
	angles = []

	for i in range(epochs):
		
		for e in range(2400//12):
			b_continue_looking = True
			
			while b_continue_looking == True:
				#print("continue",str(b_continue_looking))
				sel_idx = np.random.randint(len(csv_data))
				
				line_data = csv_data.iloc[[sel_idx]].reset_index()
				
				steering_angle = float(line_data['steering'])

				if(abs(steering_angle)<0.1):
					rand_val = np.random.uniform()
					if(rand_val>threshold):
						a_s = preprocess_data(line_data)
						b_continue_looking = False
				else:
					a_s = preprocess_data(line_data)					
					b_continue_looking = False
			angles.extend(a_s)
		threshold = 1/(i+1)
		#print("i",i)
		#print("threshold",threshold)
	return angles

def process_data_3():
	data_path = 'CAR_data/data/'
	img_path = 'IMG/'
	csv_path = 'driving_log.csv'

	csv_data = pd.read_csv(data_path+csv_path,usecols=["center","left","right","steering"])

	threshold = 0.5
	batch_size = 600
	epochs = 10
	angles = []
	sangles = []

	for i in range(epochs):
		
		for e in range(2400//4):
			b_continue_looking = True
			
			while b_continue_looking == True:
				cam_choice = np.random.randint(3)
				
				sel_idx = np.random.randint(len(csv_data))
				
				line_data = csv_data.iloc[[sel_idx]].reset_index()
				
				steering_angle = float(line_data['steering'])
				sangles.append(steering_angle)
				cam_steering_angle = cam_adjust_steering_angle(cam_choice,steering_angle)

				if((abs(steering_angle))<0.1):
					rand_val = np.random.uniform()
					if(rand_val>threshold):
						a_s = augment_steering(cam_steering_angle)
						b_continue_looking = False
					#b_continue_looking = True
				else:
					a_s = augment_steering(cam_steering_angle)					
					b_continue_looking = False
			sangles.append(steering_angle)
			sangles.append(steering_angle)
			sangles.append(steering_angle)
			sangles.append(steering_angle)			
			angles.extend(a_s)
		print(threshold)
		threshold = 0.25/(i+2)
		
	return angles,sangles

def cam_adjust_steering_angle(cam_choice,angle):
	if(cam_choice==0):
		return angle
	if(cam_choice==1):
		nangle = 0.2+angle
		return nangle
	if(cam_choice==2):
		nangle = angle-0.2
		return nangle

def process_data_4():
	data_path = 'CAR_data/data/'
	img_path = 'IMG/'
	csv_path = 'driving_log.csv'

	csv_data = pd.read_csv(data_path+csv_path,usecols=["center","left","right","steering"])

	threshold = 1
	batch_size = 600
	epochs = 10
	angles = []
	sangles = []

	for i in range(epochs):
		
		
		
		for e in range(2400):
			b_continue_looking = True
			
			while b_continue_looking == True:
				cam_choice = np.random.randint(3)
				
				sel_idx = np.random.randint(len(csv_data))
				
				line_data = csv_data.iloc[[sel_idx]].reset_index()
				
				steering_angle = float(line_data['steering'])
				sangles.append(steering_angle)
				#cam_steering_angle = cam_adjust_steering_angle(cam_choice,steering_angle)
				sa = preprocess_angle(steering_angle,cam_choice)

				if((abs(sa)<0.1)):
					rand_val = np.random.uniform()
					if rand_val> threshold+0.2:
						b_continue_looking = False
				elif((0.22>abs(sa)>0.18)):
					rand_val = np.random.uniform()
					if rand_val> threshold+0.5:
						b_continue_looking = False
				else:						
					b_continue_looking = False
			sangles.append(steering_angle)		
			angles.append(sa)
		threshold = 1/(i+1)
		print("threshold",threshold)				
	return angles,sangles

def preprocess_angle(ang,cam_choice):
	if cam_choice is not None:
		angle = cam_adjust_steering_angle(cam_choice,ang)
	else:
		angle = ang
	br_choice = np.random.randint(2)
	tr_choice = np.random.randint(2)
	fl_choice = np.random.randint(2)

	if tr_choice==1:
		angle = aug_angle_tr(angle,100,40)
	if fl_choice==1:
		angle = (angle*1)
	return angle


def generate_train(img_folder_name,csv_line_data,img_dim,batch_size,threshold):
	#img_dim[0] = height
	#img_dim[1] = width
	batch_X = np.zeros((batch_size,img_dim[0],img_dim[1],3))
	
	batch_y = np.zeros(batch_size)
	
	while 1:
		for idx in range(batch_size//12):
			#print(idx)
			#print(batch_size//12)

			b_continue_looking = True
			
			while b_continue_looking == True:
				#print("continue",str(b_continue_looking))
				sel_idx = np.random.randint(len(csv_line_data))
				line_data = csv_line_data.iloc[[sel_idx]].reset_index()
				
				steering_angle = float(csv_line_data['steering'][sel_idx])

				if(abs(steering_angle)<0.1):
					rand_val = np.random.uniform()
					if(rand_val>threshold):
						imgs,angles = preprocess_data(img_folder_name,csv_line_data,img_dim)
						b_continue_looking = False
				else:
					imgs,angles = preprocess_data(img_folder_name,csv_line_data,img_dim)					
					b_continue_looking = False
			for i in range(len(imgs)):
				#print(i)
				batch_X[idx+i] = imgs[i]
				batch_y[idx+1] = angles[i]
			yield (batch_X, batch_y)


def preprocess_data(foldername,csvline,img_dims):
	
	steering_angle = float(csvline['steering'][0])
	left_img = cv2.imread(foldername+csvline['left'][0].strip())
	left_img = cv2.cvtColor(left_img,cv2.COLOR_BGR2RGB)
	left_angle = 0.2+steering_angle

	right_img = cv2.imread(foldername+csvline['right'][0].strip())
	right_img = cv2.cvtColor(right_img,cv2.COLOR_BGR2RGB)
	right_angle = -0.2+steering_angle


	center_img = cv2.imread(foldername+csvline['center'][0].strip())
	center_img = cv2.cvtColor(center_img,cv2.COLOR_BGR2RGB)

	img_data = []
	
	img_data.append((center_img,steering_angle))
	img_data.append((left_img,left_angle))
	img_data.append((right_img,right_angle))

	imgs, angles = get_augmented_representations(img_data,img_dims)
	
	return imgs,angles

def get_augmented_representations(img_data,img_dims):
	imgs = []
	angles = []
	for element in img_data:
		angle = element[1]
		img = element [0]

		# brightness augmentation
		br_img = augment_brightness(img)
		br_img = crop_resize(br_img,img_dims[0],img_dims[1])

		# translation augmentation
		tr_img,tr_angle = augment_by_translation(img,angle,100,40)
		tr_img = crop_resize(tr_img,img_dims[0],img_dims[1])

		# flip augmentation
		fl_img,fl_angle = augment_by_flipping(img,angle)
		fl_img = crop_resize(fl_img,img_dims[0],img_dims[1])

		nimg = crop_resize(img,img_dims[0],img_dims[1])

		imgs.append(nimg)
		angles.append(angle)

		imgs.append(br_img)
		angles.append(angle)
		
		imgs.append(tr_img)
		angles.append(tr_angle)

		imgs.append(fl_img)
		angles.append(fl_angle)
	return imgs,angles


