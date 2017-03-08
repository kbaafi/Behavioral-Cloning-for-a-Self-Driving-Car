"""
	Performs preprocessing of image data, cropping out the usable area in the 
	camera information and resizing the image
"""
import numpy as np
import cv2
from bc_augmentation import *
import math

def crop_resize(img,height,width):
	"""
	Crops the image

	Args:
		img: input image
	Returns:
		img: image after transformations
	"""
	shape = img.shape
	top_crop = 70#math.floor(shape[0]/5)
	bottom_crop = shape[0]-20
	img = img[top_crop:bottom_crop,0:shape[1]]
	
	img = cv2.resize(img,(width,height),interpolation = cv2.INTER_AREA)
	
	return img

def cam_select_img_angle(cam_choice,angle,csv_line):
	"""
	Selects the data for preprocessing given the choice of camera(left,right or center)
	the angle(redundant) and the selected data row
	
	Args:
		cam_choice: selected car camera{0=center, 1=left,2 = right}
		angle: fed in steering angle
		csv_line: data row number
	Returns:
		path: path to image
		angle: augmented camera angle
	"""
	if cam_choice is not None:
		if(cam_choice==0):
			path = csv_line['center'][0].strip()
			return path,angle 
		if(cam_choice==1):
			path = csv_line['left'][0].strip()
			nangle = 0.2+angle
			return path,nangle
		if(cam_choice==2):
			path = csv_line['right'][0].strip()
			nangle = angle-0.2
			return path,nangle
	else:
		path = csv_line['center'][0].strip()
		return path,angle 

def preprocess_driving_data(img_folder_name,csv_line,img_dim):
	"""
	Select image from any of the three cameras and performs preprocessing of the selected image

	On selecting the image, this function randomly changes the brightness and translation or flips
	the image from left to right

	Args:
		img_folder_name:  location of images
		csv_line:	selected data line
		img_dim:	image dimensions
	"""
	# randomly chooses between left,right or center
	img_choice = np.random.randint(3)

	steering_angle = float(csv_line['steering'][0])

	# augment steering angle based on img_choice
	img_path,steering_angle = cam_select_img_angle(img_choice,steering_angle,csv_line)
	img_path = img_folder_name+img_path

	image = cv2.imread(img_path)
	image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

	# choice for image translation augmentation
	tr_choice = np.random.randint(2)

	# choice for brightness augmentation
	br_choice = np.random.randint(2)

	#choice for flip augmentation
	fl_choice = np.random.randint(2)

	new_img = image
	new_steering_angle = steering_angle

	if(tr_choice==1):
		# perform image augmentation by translation	
		new_img,new_steering_angle = augment_by_translation(image,steering_angle,100,40)

	if(br_choice==1):
		# perform image brightness augmentation
		new_img = augment_brightness(new_img)
		
	new_img = crop_resize(new_img,img_dim[0],img_dim[1])
	
	new_img = np.array(new_img)
	
	if fl_choice==1:
		# flip image
		new_img,new_steering_angle = augment_by_flipping(new_img,new_steering_angle)

	return new_img,new_steering_angle


def generate_data_train(img_folder_name,csv_line_data,img_dim,batch_size,threshold,yvals):
	"""
	Generates a batch of training data on the fly from the image data collected
	from cameras mounted on the car
	
	Args:
		img_folder_name		:location of images
		csv_line_data		:the drive log		
		img_dim			:dimensions of output images
		batch_size		:number of datapoints in batch
		threshold		:threshold for suppression data that skews the dataset
		yvals			:array to store steering angles for investigation
	"""
	batch_X = np.zeros((batch_size,img_dim[0],img_dim[1],3))
	
	batch_y = np.zeros(batch_size)
	
	while 1:
		for idx in range(batch_size):
			
			# randomly select a datapoint from the drivelog
			sel_idx = np.random.randint(len(csv_line_data))
			line_data = csv_line_data.iloc[[sel_idx]].reset_index()
			
			b_continue_looking = True

			while b_continue_looking == True:
				# preprocess data
				img,angle = preprocess_driving_data(img_folder_name,line_data,img_dim)
				
				# dampen occurrence of data that skews the data			
				if((abs(angle)<0.1)):
					rand_val = np.random.uniform()
					if rand_val> threshold+0.2:
						b_continue_looking = False
				elif((0.22>abs(angle)>0.18)):
					rand_val = np.random.uniform()
					if rand_val> threshold+0.5:
						b_continue_looking = False
				else:						
					b_continue_looking = False
			
			batch_X[idx] = img
			batch_y[idx] = angle
			yvals.append(angle)
			yield (batch_X, batch_y)			
