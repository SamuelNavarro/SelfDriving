import csv
import os
import cv2
import numpy as np
from math import ceil
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D


# We open the csv file with the steerings and images paths
lines = []
with open('/home/samuel/Data/BehavioralCloning/OwnData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# We pick 30 % for validations because we don't have a lot of data
train_samples, validation_samples = train_test_split(lines, test_size=0.3)
validation_samples, test_samples = train_test_split(validation_samples, test_size=0.1)



def generator(samples, batch_size=64):

    num_samples = len(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            #images = [mpimg.imread(batch[i]) for batch in batch_samples for i in range(3)]


            images =[]
            img_center, img_left, img_right = [], [], []
            steerings = []
            steer_center, steer_left, steer_right = [], [], []
            for batch in batch_samples:

                image_center = mpimg.imread(batch[0])
                image_left = mpimg.imread(batch[1])
                image_right = mpimg.imread(batch[2])
                img_center.append(image_center)
                img_left.append(image_left)
                img_right.append(image_right)

                measurement = float(batch[3])
                correction = 0.2
                steering_left = measurement + correction
                steering_right = measurement - correction
                steer_center.append(measurement)
                steer_left.append(steering_left)
                steer_right.append(steering_right)

            images.extend(img_center)
            images.extend(img_left)
            images.extend(img_right)
            steerings.extend(steer_center)
            steerings.extend(steer_left)
            steerings.extend(steer_right)

            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, steerings):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement*-1)


            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield sklearn.utils.shuffle(X_train, y_train)




batch_size=64
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


model = Sequential()
model.add(Lambda(lambda x: x/ 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((60, 25), (0,0))))
model.add(Conv2D(25, (5, 5), strides=2, activation='relu'))
model.add(Conv2D(36, (5, 5), strides=2, activation='relu'))
model.add(Conv2D(48, (5, 5), strides=2, activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.25))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, \
                    steps_per_epoch=ceil(len(train_samples) / batch_size), \
                    validation_data=validation_generator, \
                    validation_steps=ceil(len(validation_samples)/batch_size),\
                    epochs=13)



model.save('model.h5')

test_generator = generator(test_samples, batch_size=batch_size)
test_loss = model.evaluate_generator(test_generator, steps=128)
print("Test loss: ", test_loss)
