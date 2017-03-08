**Behavioral Cloning Project**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png 
[image2]: ./examples/centre.png
[image3]: ./examples/left.png
[image4]: ./examples/right.png
[image5]: ./examples/aug_orig.png
[image6]: ./examples/br.png
image7]: ./examples/br.png
image8]: ./examples/br.png
image9]: ./examples/tr.png
image10]: ./examples/fl.png
image11]: ./examples/resized.png
[image12]: ./examples/orignal_steering.png
image13]: ./examples/curated_steering.png

## Rubric [Points](https://review.udacity.com/#!/rubrics/432/view)

---
###Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.json
```

####3. Submission code is usable and readable

The `model.py` file contains the code describing the architecture of the network(implemented in Keras). The file `bc_train.py` contains the pipeline that trains the model to mimic the behavior of a human driver.

The file `drive.py` emits a control signal (throttle and steering angle) based on the camera input from the simulator while the car is running. `drive.py` uses the saved model after training `model.json` and the associated weights `model.hdf5`. The output of the model is fed back to the simulator to control the car

###Model Architecture and Training Strategy

I used the model described in the Nvidia publication, [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316). The architecture is shown below:

Each layer is connected to a relu activation layer(not shown in the diagram)

![alt text][image1]

####2. Reducing overfitting

Tests conducted revealed that drop out as a complexity reduction strategy does not really work with the model. Various drop out layers were placed in different portions and in one case at the end of each convolution layer but did not give the desired result. Surprisingly what worked was training for a few epochs(6 in this case). That's one of the interesting points of this project.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model.py` line 69).


####4.Model Architecture and Training Strategy

###1. Solution Design Approach

I started by using the Nvidia model as described in the publication and built a Keras neural network based on that. The input image was of shape `(66,220,3)` I used a data generator to randomly take 24000 data points in batches of 240 to train the data. After realising that drop out does not give the desired results in the case of this model, I decided to train for a minimal number of epochs(6 in this case). 

The final network has input image of shape `(64,64,3)`

The data generator works as follows:
1. Pick a datarow randomly from the drivelog
2. Randomly select if center,left or right image
3. Based on selected image adjust the steering angle according to give the effect that the camera is still situated in the center(this creates a few problems that will be discussed later)
4. Randomly perturb the data by adding changes in brightness, translation(horizontal and vertical) and flipping the image
5. Adjust the occurrence of the steering angles such that we do not learn too much from a skewed dataset while keeping in mind that the car must drive mostly in a straight line and be able to make the necessary adjustments on soft and hard turns


###2. Creation of the Training Set & Training Process

The training data can be derived by driving the car around in the simulator. I rode around the track a few times but was not satisfied with the data I was generating. I ended up using the data provided by Udacity. A sample of the images is shown below

![alt text][image5]

I decided to utilize the left and right cameras for simulating recovery from the edge of the road. In doing this a value of 0.2 was added to the steering angle when the left camera was used and a value of 0.2 was subtracted from the camera angle when a right camera was chosen. This way images from the right and left cameras can be used as if they are from the center camera. 

The below shows the left, right and center images for one data point

![alt text][image2]

![alt text][image3]

![alt text][image4]

Augmentation is not strictly performed but the data is pertubed at random for variations in brightness, slight translation(horizontal and vertical) and flipping

Samples of perturbed images are shown below

![alt text][image5]

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]

![alt text][image10]

After the transformations each image is cropped to reveal only the portion of the road that we are interested in. The function `crop_resize` in `bc_preprocessing.py` handles this task. The resulting image is shown below:

![alt text][image11]

It must be stated that the driving data features majority of the steering angles being close to zero since a car moves in a straight almost all the time. If our data generator picks randomly a data row from the drive log, then we run the risk of collecting data with a very large majority having steering angle close to 0. Even if we choose some images from the left or right camera we can also have a situation where a disproportionate number of examples occur at the points around 0, +0.2 and -0,2. A method for throttling the number of examples with steering angles around 0, -0.2 and +0.2. Ideally since the car runs in a straight line most of the time we want more of the straight line data than any other category, however we also need to be able to make good decisions during sharp turns, this is where the left and right camera images become useful. In plain english this model aims to `drive in a straight line as much as possible, when a turn is reached make the necessary adjustment, after that get back into the straight line`. Lines 143 to 151 in `bc_preprocessing.py` shows how this is done. 

When an image, angle pair is generated in the generator, if the steering angle after augmentation is around 0 or -0.2 or +0.2, we decide whether or not to reject the data and keep looking for a better candidate. This way we are able to force the correct distribution of data onto the model.

The plot of the distributions before and after data selection is shown below

![alt text][image13]

![alt text][image13]


The performance of the network at driving the car is shown in [this video link](https://www.youtube.com/watch?v=LNmAZbp9z3M&t=503s)


