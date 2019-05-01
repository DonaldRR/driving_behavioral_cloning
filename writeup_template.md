# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

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

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
