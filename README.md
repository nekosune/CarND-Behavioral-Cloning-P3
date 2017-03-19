

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

—-
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* This README.md writing up the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
sh
python drive.py model.h5

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64, Basically the NVIDIA model, with an added noise layer (model.py lines 86-95) 

The model  is normalized in the model using a Keras lambda layer (code line 77). 

####2. Attempts to reduce overfitting in the model

The model has a noise layer to help prevent over-fitting. (model.py lines 83). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 49-51). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 96).

####4. Appropriate training data

I used center lane driving, multiple times in each direction around the track, as well as some recoverys both ways, to make the training Data

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I used center lane driving, multiple times in each direction around the track, as well as some recoverys both ways, to make the training Data

I used the NVIDIA neural net, as it was a relatively simple model, that had been shown to do the job.
I had tried inception v4, a varient of, to see if it would work, but it created an overly complicated model, that did not produce as good results.

I had at first a large amount of validation error , so I played around with the number of epochs, as well as increasing data with flipping, and manually increasing the data by driving more. I also aded the guassian noise layer to help prevent any overfitting that was helping to cause this

it took quite a few attempts to get a model that didnt go off the road and onto the dirt, when it splits from road to dirt track. This turned out due to me not randomising before I split, but after, so the validation set , ie the last 20% of the track, was never taught.

####2. Final Model Architecture
The final model architecture (model.py lines 86-95) consisted of a convolution neural network with the following layers and layer sizes 
Convolution - 24 - 5x5 subsample - 2x2
Convolution - 36 - 5x5 subsample - 2x2
Convolution - 48 - 5x5 subsample - 2x2
Convolution -  64- 3x3 
Convolution -  64- 3x3 
Flatten
Dense - 100
Dense - 50
Dense - 10
Dense - 1



Here is a visualization of the architecture


![archetcture][arch.png]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded five laps on track one using center lane driving. 

I then recorded some recoveries, so the vehicle woudl learn to go towards the centre 
I then recorded five laps going the opposite direction to get more data points, and to help prevent overfitting


To augment the data sat, I also flipped images and angles thinking that this would ... 
After the collection process,  preprocessed this data by ... Normalizing and cropping it.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by validation tending to go up after that amount I used an adam optimizer so that manually training the learning rate wasn't necessary.
