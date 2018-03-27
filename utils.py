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
This provides utility functions to easily retrieve the training and testing
data as used by this project's image classifier. Note that all images
retrieved by these functions are resized to be 300 x 300 pixels.
"""

import numpy as np
from skimage import io, transform


def get_data(img_root, x_file, y_file):
    """
    Retrieves the image data and labels from the directory
    and files as given by the associated parameters. Note that
    images retrieved are resized to be 300 x 300 pixels.

    :param img_root: The directory where the images are located.
    :param x_file: The name of the file with the names of the images.
    :param y_file: The name of the file with the labels for the images.
    :returns: A numpy array with the image data and a numpy array
              with the labels.
    """

    x_s = np.loadtxt(x_file, dtype=str)
    y_s = np.loadtxt(y_file)

    tempx = []
    tempy = []
    for i, xname in enumerate(x_s, 0):
        img_name = img_root + xname
        image = transform.resize(io.imread(img_name), (300, 300))
        tempx.append(image)
        tempy.append(y_s[i])

    return np.array(tempx), np.array(tempy)


def get_test_data():
    """
    Retrieves the test image data and labels. Note that images retrieved
    are resized to be 300 x 300 pixels.

    :returns: A numpy array with the test image data and a numpy array
              with the test labels.
    """

    return get_data('testset/test/', 'testset/test-x', 'testset/test-y')


def get_train_data():
    """
    Retrieves the training image data and labels. Note that images
    retrieved are resized to be 300 x 300 pixels.

    :returns: A numpy array with the training image data and a numpy
              array with the training labels.
    """

    return get_data('dataset/train', 'dataset/train-x', 'dataset/train-y')


def print_baseline(labels):
    """
    Prints the accuracy if we just classified all images one label for 0 and 1

    :param labels: The actual classification labels for the data set as a
                   numpy array.
    """

    numzeros = 0.0
    numones = 0.0
    for i in labels:
        if i == 0.0:
            numzeros += 1.0
        if i == 1.0:
            numones += 1.0

    print('Guess Zero: ', str(numzeros / float(len(labels))))
    print('Guess One: ', str(numones / float(len(labels))))
