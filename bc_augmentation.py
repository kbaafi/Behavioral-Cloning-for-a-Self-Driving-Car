"""
	Performs various augmentation functions on a given data set(image and 
	steering angle) and returns the augmented image and resulting angle 
	(if it changes) 
"""

import numpy as np
import cv2

def augment_brightness(image):
	"""
	Adjusts the brightness of an image by scaling the V channel by a random
	number
	Args:
		image: image to be adjusted
	Returns:
		new_image: the transformed image
	"""
	#print("TEMP_TYPE",type(image))
	temp_img = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
	brightness_adjustment = np.random.uniform()
	temp_img[:,:,2] = np.array(temp_img[:,:,2]*brightness_adjustment)
	#print("TEMP_TYPE",type(temp_img))
	new_image = cv2.cvtColor(temp_img,cv2.COLOR_HSV2RGB)

	return new_image

def augment_by_translation(image,steering_angle,x_translation_range,y_translation_range):
	"""
	Translates given image by a random value within the range (0,translation_range)
	Since the translation has an effect on the steering angle a new steering
	angle will have to be calculated
	Args:
		image: image to be translated
		steering_angle: original steering angle associated with image
		x_translation_range: maximum amount of translation desired in the x-axis
		y_translation_range: maximum amount of translation desired in the y-axis
	Returns:
		new_image: translated image
		new_steering_angle: modified steering angle after translation
	"""
	x_translation = (x_translation_range*np.random.uniform())-(x_translation_range/2)
	y_translation = (y_translation_range*np.random.uniform())-(y_translation_range/2)
	
	new_steering_angle = steering_angle+((x_translation/x_translation_range)*2)*0.2
	
	trans_matrix = np.float32([[1,0,x_translation],[0,1,y_translation]])

	new_image = cv2.warpAffine(image,trans_matrix,(image.shape[1],image.shape[0]))
	
	return new_image,new_steering_angle

def augment_by_flipping(image,steering_angle):
	"""
	Flips the image horizontally and reverses the steering angle
	Args:
		image: image to be flipped
		steering_angle: original steering angle associated with image
	Returns:
		new_image: translated image
		new_steering_angle: modified steering angle after flipping
	"""
	new_image = np.fliplr(image)
	new_steering_angle = -steering_angle
	return new_image,new_steering_angle

