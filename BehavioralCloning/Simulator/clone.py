import csv
import cv2
import numpy as np

lines = []
with open('/home/samuel/Data/BehavioralCloning/OwnData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images =[]
img_center, img_left, img_right = [], [], []


steerings = []
steer_center, steer_left, steer_right = [], [], []
for line in lines:
    image_center = cv2.imread(line[0])
    image_left = cv2.imread(line[1])
    image_right = cv2.imread(line[2])
    img_center.append(image_center)
    img_left.append(image_left)
    img_right.append(image_right)

    measurement = float(line[3])
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


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D


model = Sequential()
model.add(Lambda(lambda x: x/ 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0,0))))

model.add(Conv2D(25, (5, 5), strides=2, activation='relu'))
model.add(Conv2D(36, (5, 5), strides=2, activation='relu'))
model.add(Conv2D(48, (5, 5), strides=2, activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=5)


model.save('model.h5')


