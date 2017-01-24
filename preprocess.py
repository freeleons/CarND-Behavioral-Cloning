import pandas as pd
import numpy as np
from scipy.misc import imread
import pickle
from sklearn.cross_validation import train_test_split

driving_log = pd.read_csv('driving_log.csv', index_col=False)
driving_log.columns = ['Center', 'Left', 'Right', 'Steering Angle', 'Throttle', 'Break', 'Speed']

# 2. Prepare Data for Generator
X_train = []
y_train = []
for index, content in driving_log.iterrows():

    print(index)
    print(content['Center'])
    print(content['Steering Angle'])
    X_train.append(imread(content['Center'].strip()))
    y_train.append(content['Steering Angle'])
    X_train.append(imread(content['Left'].strip()))
    y_train.append(content['Steering Angle'])
    X_train.append(imread(content['Right'].strip()))
    y_train.append(content['Steering Angle'])

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_train,
    y_train,
    test_size=0.10,
    random_state=497376801)

with open('data.pickle', 'wb') as f:
    pickle.dump(
                    {
                        'X_train': X_train_split,
                        'y_train': y_train_split,
                        'X_test': X_test_split,
                        'y_test': y_test_split
                    },
                    f, pickle.HIGHEST_PROTOCOL)
f.close()
