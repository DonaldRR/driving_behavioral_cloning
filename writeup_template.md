# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

[//]: # (Image References)

[steering_angles]: ""
[steering_angles_non_zero]: ""
[ori_images]: ""
[cropped_images]: ""
[flipped_images]: ""


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

### Training data generation and Preprocessing

#### 1. Generate data by hands

More data is always better. Though some training data is provided in this project, I ran the simulator to generate more training data to make the resulting model more robust to different situations. During the "driving" process, I tried to steer my car close to left and right lane lines and steer back trying to teach it how to get back to the middle of lane when the car comes into such situation. Roughly I drove 10 rounds.

#### 2. Steering angle data analysis

Let's have a look at how the steering angles distributed in the sata. 

![alt text][steering_angles]

Above histogram shows a large number of data is centered at value zero. So I did the statistics to find how many zero-value angles and non-zeros in this training. It turns out that there are `15910` and `4642` zero and nonzero values. The distribution of nonzeros are shown below.

![alt text][steering_angles_non_zero]

It's clear that much of them are negative and this result is intuitive since the car drives counter-clockwise.

In a sense, the training steering angle data is bias. This problem will be considered and treated in the following preprocessing part.

#### 3. Image data analysis

Here are images from the left, middle and right cameras respectively. 

![alt text][ori_images]

We know that, the appropriate steering angle can be obtained from only the lower part of the images, i.e. the lane, not the scene above. So each image is cropped (shown below).

![alt text][cropped_images]

This approach can help the model to generalize on future data.

#### 4. Data preprocessing

The first step is data augmentation.

Since the training data is bias as mentioned above. This step flips each image horizontally and assigns it the steering angle of the original one multiplied by -1. The following image is the image from the middle camera and the flipped image.

![alt text][flipped_images]

Each data item in the log CSV file has 3 images from different cameras and one corresponding steering angle. In order to utilze images from left and right cameras, I attepmpted to assign steering angles to them. Assume the steering angle of the center image is `a`, its left and right correspondings are `a+0.2` and `a-0.2`. The 0.2 is used as the offset to teach the car to drive back to the center of lane manually.


### Model Architecture and Training Strategy

#### 1. Neural Network Architecture (referenced from [link](https://devblogs.nvidia.com/deep-learning-self-driving-cars/))

**My neural network model has the following layers:**

+ Convolution layer (filters: 3, kernel size: (5, 5), strides: (2, 2), padding='valid')
+ Convolution layer (filters: 24, kernel size: (5, 5), strides: (2, 2), padding='valid')
+ Convolution layer (filters: 36, kernel size: (5, 5), strides: (2, 2), padding='valid')
+ Convolution layer (filters: 48, kernel size: (3, 3), strides: (2, 2), padding='valid')
+ Convolution layer (filters: 64, kernel size: (3, 3), strides: (2, 2), padding='valid')
+ Flatten layer
+ Fully connected layer (units: 100)
+ Fully connected layer (units: 50)
+ Fully connected layer (units: 10)
+ Fully connected layer (units: 1)

Each layer except flatten layer and the last layer is followed by batchnormalization layer and ReLU activation implicitly.

#### 2. Attempts to reduce overfitting in the model

Here are several attempts:

+ BatchNormalization: The model contains multiple batchnormalization layers to combat overfitting.
+ Error metric: Mean squared error is used as the error metric other than the mean absolute error.
+ Validation: Validation data is separated with training data and the early stopping callback monitoring validation error only stops training when validation error doesn't decrease.

#### 3. Model parameter tuning

The Adam optimizer is deployed in the training process.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 
