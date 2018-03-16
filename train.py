# Copyright 2018 Dan Tran
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Trains an image classifier model on the training set and saves the model
to a file, outputing the accuracy of the model on the development set
along with the baseline accuracies. This uses Keras with a TensorFlow
backend to construct, fit, and evaluate the model.
"""

import os
import random as rn

from keras import backend as k
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import numpy as np
from skimage import io, transform
import tensorflow as tf

from utils import print_baseline


def seed_randomness():
    """
    Ensure we get reproducable results by seeding at all relevant places
    and ensure we are only running on one thread for TensorFlow
    """

    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(42)
    tf.set_random_seed(42)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    k.set_session(sess)


def load_data_with_dev():
    """
     Retrieves image and label data from the training set and carve out
     a development set from 10% of the training set.

     :returns: A numpy array with the training image data, a numpy array
               with the training image labels, a numpy array with the
               development image data, a numpy array with the development
               image labels, and a dict with the class weights.
    """

    # Load in picture paths and labels
    root = 'dataset/'
    train_x = np.loadtxt(root + 'train-x', dtype=str)
    train_y = np.loadtxt(root + 'train-y')

    # Shuffle indices
    idx = list(range(len(train_x)))
    np.random.shuffle(idx)

    # Create development set indices
    devidx = {-1}
    for i in range(int(len(train_x) / 10)):
        r = -1
        while r in devidx:
            r = np.random.randint(len(train_x))
        devidx.add(r)
    devidx.remove(-1)

    # Load in picture data into training and development sets
    tempx = []
    tempy = []

    tempdevx = []
    tempdevy = []

    class_weight_temp = {0:  0., 1: 0.}

    for i, x in enumerate(idx, 0):
        img_name = root + 'train/' + train_x[x]
        image = transform.resize(io.imread(img_name), (300, 300))
        if i in devidx:
            tempdevx.append(image)
            tempdevy.append(train_y[x])
        else:
            tempx.append(image)
            tempy.append(train_y[x])
            class_weight_temp[train_y[x]] += 1.0

    class_weight = {0: class_weight_temp[1], 1: class_weight_temp[0]}

    trainx = np.array(tempx)
    trainy = np.array(tempy)

    devx = np.array(tempdevx)
    devy = np.array(tempdevy)

    return trainx, trainy, devx, devy, class_weight


def construct_model():
    """
    Constructs the model that will be used to classify images.

    :returns: The model.
    """

    model = Sequential()

    model.add(Conv2D(8, 9, input_shape=(300, 300, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(20, 9, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, 9, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def main():
    # Make sure we get reproducable results
    # Can be commented out if we want faster performance
    seed_randomness()

    # Load the training and development data and calculate the class weights
    trainx, trainy, devx, devy, class_weight = load_data_with_dev()

    # Construct the model
    model = construct_model()

    # Train the model with the train set
    model.fit(trainx, trainy, epochs=15, batch_size=32, class_weight=class_weight)

    # Evaluate the model on the development set
    score = model.evaluate(devx, devy, batch_size=32)

    # Report findings
    print('Finished Training')
    print('Dev Acuracy: ', score[1])
    model.save('model.ker')

    # Compare if we just guess one or zero
    print_baseline(devy)


if __name__ == '__main__':
    main()
