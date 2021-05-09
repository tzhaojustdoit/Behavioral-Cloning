import csv
import numpy as np
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers.pooling import MaxPool2D
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense, Lambda, Dropout, Flatten, Cropping2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from scipy import ndimage

data_dir = '/opt/carnd_p3/mydata/'
lines = []
with open(data_dir + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.2
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for line in batch_samples:
                source_paths = line[:3]
                for i in range(3):
                    filename = source_paths[i].split('/')[-1]
                    current_path = data_dir + 'IMG/' + filename
                    image = ndimage.imread(current_path)
                    if image is None:
                        continue
                    images.append(image)
                    measurement = float(line[3])
                    if i == 1:
                        measurement += correction
                    elif i == 2:
                        measurement -= correction
                    measurements.append(measurement)
                    # flip
                    image_flipped = np.flip(image)
                    measurement_flipped = -measurement
                    images.append(image_flipped)
                    measurements.append(measurement_flipped)
            X_train = np.array(images)
            y_train = np.array(measurements, dtype=np.float)
            res = (X_train, y_train)
            yield res

batch_size = 100

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# model architecture
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=ceil(
                                         len(train_samples)/batch_size),
                                     validation_data=validation_generator,
                                     validation_steps=ceil(
                                         len(validation_samples)/batch_size),
                                     epochs=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
