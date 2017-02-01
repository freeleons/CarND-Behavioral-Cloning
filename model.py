import numpy as np
import pickle
import json
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2
from keras.layers.core import SpatialDropout2D, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D


training_file = './data.pickle'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X_train, y_train = train['X_train'], train['y_train']
X_test, y_test = train['X_test'], train['y_test']

# def train_data_generator(X_train_arg, y_train_arg):
#     for i in range(len(X_train_arg)):
#         # yield ({'input': X_train_arg[i]}, {'output': y_train_arg[i]})
#         yield (np.array([X_train_arg[i]]), np.array([y_train_arg[i]]))

def train_data_generator(X_train_arg, y_train_arg):
    while True:
        for i in range(len(X_train_arg)):
            yield (np.array([X_train_arg[i]]).reshape([-1,160,320,3]), np.array([y_train_arg[i]]))


num_of_train = len(X_train)
num_of_angle = len(y_train)
num_of_test = len(X_test)
num_of_test_angle = len(y_test)
image_shape = X_train[0].shape

print("Image Shape:", image_shape)
print("Number of Training Examples:", num_of_train)
print("Number of Training Angles:", num_of_angle)
print("Number of Testing Examples:", num_of_test)
print("Number of Testing Angles:", num_of_test_angle)

input_shape = X_train[0].shape
print(input_shape, 'input shape')

batch_size = 64 # The lower the better
nb_classes = 1 # The output is a single digit: a steering angle
nb_epoch = 5 # The higher the better

def resize(X):
    # import tensorflow here so module is available when recreating pipeline from saved json.
    import tensorflow
    return tensorflow.image.resize_images(X, (40, 160))

# import model and wieghts if exists
try:
	with open('model.json', 'r') as jfile:
	    model = model_from_json(json.load(jfile))

	# Use adam and mean squared error for training
	model.compile("adam", "mse")

	# import weights
	model.load_weights('model.h5')

	print("Imported model and weights")

# If the model and weights do not exist, create a new model
except:
	# If model and weights do not exist in the local folder,
	# initiate a model

	# number of convolutional filters to use
	nb_filters1 = 16
	nb_filters2 = 8
	nb_filters3 = 4
	nb_filters4 = 2

	# size of pooling area for max pooling
	pool_size = (2, 2)

	# convolution kernel size
	kernel_size = (3, 3)

	# Initiating the model
	model = Sequential([
    # Preprocess
    # Crop above horizon and car hood to remove uneeded information
    # Resize images to improve performance
    # Normalize to keep weight values small with zero mean, improving numerical stability.
    Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)),
    Lambda(resize),
    BatchNormalization(axis=1),
    # Conv 5x5
    Convolution2D(24, 5, 5, border_mode='same', activation='elu'),
    MaxPooling2D(border_mode='same'),
    SpatialDropout2D(0.2),
    # Conv 5x5
    Convolution2D(36, 5, 5, border_mode='same', activation='elu'),
    MaxPooling2D(border_mode='same'),
    SpatialDropout2D(0.2),
    # Conv 5x5
    Convolution2D(48, 5, 5, border_mode='same', activation='elu'),
    MaxPooling2D(border_mode='same'),
    SpatialDropout2D(0.2),
    # Conv 3x3
    Convolution2D(64, 3, 3, border_mode='same', activation='elu'),
    MaxPooling2D(border_mode='same'),
    SpatialDropout2D(0.2),
    # Conv 3x3
    Convolution2D(64, 3, 3, border_mode='same', activation='elu'),
    MaxPooling2D(border_mode='same'),
    SpatialDropout2D(0.2),
    # Flatten
    Flatten(),
    # Fully Connected
    Dense(100, activation='elu', W_regularizer=l2(1e-6)),
    Dense(50, activation='elu', W_regularizer=l2(1e-6)),
    Dense(10, activation='elu', W_regularizer=l2(1e-6)),
    Dense(1)
    ])

# Print out summary of the model
model.summary()

# Compile model using Adam optimizer
# and loss computed by mean squared error
model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['accuracy'])

### Model training
# history = model.fit(X_train, y_train,
                    # batch_size=batch_size, nb_epoch=nb_epoch,
                    # verbose=1, validation_split = 0.1)

history = model.fit_generator(train_data_generator(X_train, y_train), samples_per_epoch = len(X_train), nb_epoch=nb_epoch, verbose=1)
score = model.evaluate(np.array([X_test]).reshape([-1,160,320,3]), np.array([y_test]).reshape(-1, 1), verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

import json
import os
import h5py

# Save the model.
# If the model.json file already exists in the local file,
# warn the user to make sure if user wants to overwrite the model.
if 'model.json' in os.listdir():
	print("The file already exists")
	print("Want to overwite? y or n")
	user_input = input()

	if user_input == "y":
		# Save model as json file
		json_string = model.to_json()

		with open('model.json', 'w') as outfile:
			json.dump(json_string, outfile)

			# save weights
			model.save_weights('./model.h5')
			print("Overwrite Successful")
	else:
		print("the model is not saved")
else:
	# Save model as json file
	json_string = model.to_json()

	with open('model.json', 'w') as outfile:
		json.dump(json_string, outfile)

		# save weights
		model.save_weights('./model.h5')
		print("Saved")
