import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, GaussianNoise, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
lines = []
images = []
measurements = []

# Function to split the array
def split(arr, size):
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr = arr[size:]
    arrs.append(arr)
    return arrs
# load the driving log
with open('data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        steering_center = float(line[3])
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        source_path = line[0]
        source_path_left = line[1]
        source_path_right = line[2]
        filename = source_path.split('\\')[-1]
        filename_left = source_path_left.split('\\')[-1]
        filename_right = source_path_right.split('\\')[-1]
        current_path = 'data2/IMG/'+filename
        current_path_left = 'data2/IMG/'+filename_left
        current_path_right = 'data2/IMG/'+filename_right
        images.append(current_path)
        images.append(current_path_left)
        images.append(current_path_right)
        measurements.append(steering_center)
        measurements.append(steering_left)
        measurements.append(steering_right)
# Split the randomized log into 80% and 20%
linked = list(zip(images, measurements))
random.shuffle(linked)
splitList = split(linked, int(len(images)*0.8))


# Generic generator, will return each pass, the image, and flipped image
def generator(listToUse):
    while 1:
        images2 = list(listToUse)
        random.shuffle(images2)
        while len(images2) > 0:
            current = images2.pop()
            image = cv2.imread(current[0])
            measurement = current[1]
            augmented_images, augmented_measurements = [], []
            augmented_images.append(image)
            augmented_measurements.append(measurement)
            augmented_images.append(cv2.flip(image, 1))
            augmented_measurements.append(measurement*-1.0)
            yield (np.array(augmented_images), np.array(augmented_measurements))




# Start of model
model = Sequential()

# Lambda normalization
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

# Croping the layer to relevent parts
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# Adding some noise to prevent over-fitting
model.add(GaussianNoise(0.1))

# Main neural net
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(generator(splitList[0]), validation_data=generator(splitList[1]), samples_per_epoch=len(splitList[0])*2, nb_val_samples=len(splitList[1])*2, nb_epoch=3, verbose=1)
print(history_object.history.keys())
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('history.png')
model.save("modelnewData.h5")
