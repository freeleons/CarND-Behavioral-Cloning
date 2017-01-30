import numpy as np
import pickle
import json
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop


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
            yield (np.array([X_train_arg[i]]), np.array([y_train_arg[i]]))


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
nb_epoch = 10 # The higher the better

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
	model = Sequential()

	# Starting with the convolutional layer
	# The first layer will turn 1 channel into 16 channels
	model.add(Convolution2D(nb_filters1, kernel_size[0], kernel_size[1],
	                        border_mode='valid',
	                        input_shape=(input_shape)))
	# Applying ReLU
	model.add(Activation('relu'))
	# The second conv layer will convert 16 channels into 8 channels
	model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1]))
	# Applying ReLU
	model.add(Activation('relu'))
	# The second conv layer will convert 8 channels into 4 channels
	model.add(Convolution2D(nb_filters3, kernel_size[0], kernel_size[1]))
	# Applying ReLU
	model.add(Activation('relu'))
	# The second conv layer will convert 4 channels into 2 channels
	model.add(Convolution2D(nb_filters4, kernel_size[0], kernel_size[1]))
	# Applying ReLU
	model.add(Activation('relu'))
	# Apply Max Pooling for each 2 x 2 pixels
	model.add(MaxPooling2D(pool_size=pool_size))
	# Apply dropout of 25%
	model.add(Dropout(0.25))

	# Flatten the matrix. The input has size of 360
	model.add(Flatten())
	# Input 360 Output 16
	model.add(Dense(16))
	# Applying ReLU
	model.add(Activation('relu'))
	# Input 16 Output 16
	model.add(Dense(16))
	# Applying ReLU
	model.add(Activation('relu'))
	# Input 16 Output 16
	model.add(Dense(16))
	# Applying ReLU
	model.add(Activation('relu'))
	# Apply dropout of 50%
	model.add(Dropout(0.5))
	# Input 16 Output 1
	model.add(Dense(nb_classes))

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
score = model.evaluate(X_test, y_test, verbose=0)
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
