import json
import pandas as pd
import numpy as np
from scipy.misc import imread
from sklearn.cross_validation import train_test_split
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2
from keras.layers.core import SpatialDropout2D, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D




driving_log = pd.read_csv('driving_log.csv', index_col=False)
driving_log.columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Break', 'Speed']

# 2. Prepare Data for Generator
# X_train = np.array([]).reshape([-1,160,320,3])
# X_train = np.array([])
# y_train = np.array([])

X_train = []
y_train = []
for index, content in driving_log.iterrows():

    print(index)
    print(content['Center'])
    print(content['Steering'])

    angle = content['Steering']
    center = angle

    X_train.append(content['Center'].strip())
    y_train.append(center)

    # angle = content['Steering']
    # left = angle + 0.1
    # right = angle - 0.1
    #
    # X_train.append(content['Left'].strip())
    # y_train.append(left)
    #
    # X_train.append(content['Right'].strip())
    # y_train.append(right)

X_train, X_test_split, y_train, y_test = train_test_split(
    X_train,
    y_train,
    test_size=0.001,
    random_state=3)

X_test = []
for image in X_test_split:
    print('Test:',image)
    X_test.append(imread(image))


#Write a generator to feed the data to the model to save memory
def train_data_generator(X_train_arg, y_train_arg):
    while True:
        for i in range(len(X_train_arg)):
            yield (np.array(imread(X_train_arg[i])).reshape([-1,160,320,3]), np.array([y_train_arg[i]]))


num_of_train = len(X_train)
num_of_angle = len(y_train)
num_of_test = len(X_test)
num_of_test_angle = len(y_test)

print("Number of Training Examples:", num_of_train)
print("Number of Training Angles:", num_of_angle)
print("Number of Testing Examples:", num_of_test)
print("Number of Testing Angles:", num_of_test_angle)


nb_epoch = 1

def resize(X):
    import tensorflow
    return tensorflow.image.resize_images(X, (40, 160))



model = Sequential()
# Crop the horizon and car hood to remove the useless information
model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)))
# Resize the images to 40X160 to improve performance
model.add(BatchNormalization(axis=1))
# Convolutional 5x5
model.add(Convolution2D(24, 5, 5, border_mode='same', activation='relu'))
model.add(MaxPooling2D(border_mode='same'))
model.add(SpatialDropout2D(0.2))
# Convolutional 5x5
model.add(Convolution2D(36, 5, 5, border_mode='same', activation='relu'))
model.add(MaxPooling2D(border_mode='same'))
model.add(SpatialDropout2D(0.2))    # Conv 5x5
model.add(Convolution2D(48, 5, 5, border_mode='same', activation='relu'))
model.add(MaxPooling2D(border_mode='same'))
model.add(SpatialDropout2D(0.2))
# Convolutional 3x3
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(border_mode='same'))
model.add(SpatialDropout2D(0.2))
# Convolutional 3x3
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(border_mode='same'))
model.add(SpatialDropout2D(0.2))

model.add(Flatten())
# Fully Connected Layer
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


# Printout the summary of the model
model.summary()

# Compile model using Adam optimizer and loss computed by mean squared error
model.compile(loss='mse',
              optimizer=Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              metrics=['accuracy'])

# Training the Model
history = model.fit_generator(train_data_generator(X_train, y_train), samples_per_epoch = len(X_train), nb_epoch=nb_epoch, verbose=1)
score = model.evaluate(np.array([X_test]).reshape([-1,160,320,3]), np.array([y_test]).reshape(-1, 1), verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

import json
import os
import h5py

# Save the model. Save the model as json file
json_string = model.to_json()
with open('model.json', 'w') as outfile:
	json.dump(json_string, outfile)
	# save weights
	model.save_weights('./model.h5')
	print("Saved")
