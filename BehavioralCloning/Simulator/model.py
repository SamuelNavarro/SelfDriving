import csv
import os
import cv2
import numpy as np
from math import ceil
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

# We open the csv file with the steerings and images paths
lines = []
with open('/home/samuel/Data/BehavioralCloning/OwnData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# We pick 30 % for validations because we don't have a lot of data
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
validation_samples, test_samples = train_test_split(validation_samples, test_size=0.1)


def generator(samples, batch_size=64):
    """Generates data for training to avoid loading all data in memory
    :param samples: The data is directly from the driving_log csv file
    :param batch_size: Number of lines to be processed.

    :returns: Generator with batch of lines to be feed into model
    :Example:
    >>> train_generator = generator(train_samples, batch_size=batch_size)
    """

    num_samples = len(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            # center, left, right
            images = [mpimg.imread(batch[i]) for batch in batch_samples for i in range(3)]
            # Same order as images: center, left, right
            measurements = [(float(batch[3]), float(batch[3]) + 0.2, float(batch[3]) - 0.2)
                           for batch in batch_samples]
            measurements = np.array(measurements)
            measurements = measurements.ravel()
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
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

# We used roughly the same architecture as NVIDIA from:
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()
model.add(Lambda(lambda x: x/ 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((60, 25), (0,0))))
model.add(Conv2D(24, (5, 5), strides=2, activation='relu'))
model.add(Conv2D(36, (5, 5), strides=2, activation='relu'))
model.add(Conv2D(48, (5, 5), strides=2, activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')

checkpoint = ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True)
stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4)



history = model.fit_generator(train_generator, \
                    steps_per_epoch=ceil(len(train_samples) / batch_size), \
                    validation_data=validation_generator, \
                    validation_steps=ceil(len(validation_samples)/batch_size),\
                    epochs=20, verbose=1, callbacks=[checkpoint, stopper])


#model.save('model.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model MSE Loss')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Validation Set'], loc='upper right')
plt.savefig('loss.png')

plot_model(model, to_file='model_plot.png', show_shapes=True)

test_generator = generator(test_samples, batch_size=batch_size)
test_loss = model.evaluate_generator(test_generator, steps=64)
print("Test loss: ", test_loss)
