# Van Gogh Classifier
A machine learning project using Keras with a TensorFlow backend to classify paintings as either being created by Vincent van Gogh or not.

## Building
This project was built using Python 3.6 with the numpy, scikit-image, Keras, and TensorFlow libraries. Simply install those packages using pip and execute the desired script.

## Python Files
* `train.py`: Contains the script used to train the classifier. Outputs a `model.ker` file that is the trained model.
* `test.py`: Contains the script used to test the classifier. Requires a `model.ker` file that is the model we want to test.
* `utils.py`: Contains useful utility functions, like loading training or testing data and other miscellanious tasks.

## Dataset
The dataset consists of paintings from the 13th to the 20th century by various European and Russian artists, including paitings by Vincent van Gogh. The data has been randomly split into two directories: the `dataset/` directory and the `testset/` directory.  The `dataset/` directory refers to the training set in which the model is supposed to use to learn and train a classifier.  The `testset/` directory referst to the testing set in which the model is supposed to use to evaluate its prediction capabilities.

In each of those directories, there are two files and another directory. The other directory, either `train/` or `test/` for `dataset/` and `testset/` respectively, are where the actual images are located.  The first file, either `train-x` or `test-x` for `dataset/` and `testset/` respectively, is a plain textfile with a list of filenames, one filename per line, which corresponds to the filenames of the images in the other directory.  The second file, either `train-y` or `test-y` for `dataset/` and `testset/` respectively, is a plain textfile with a list of labels, one label per line, which corresponds to wether or not an image was painted by Vincent van Gogh or not, with `1` being the image is painted by Vincent van Gogh are `0` being it wasn't.  The first label corresponds to the first filename, the second label corresponds to the second filename, and so on.

## Methodology
This project utilized a convolutional neural network model with three convolutional layers, each having a max pooling layer right after, having a dropout layer to prevent overfitting, and then four fully connected layers afterwards.  ReLU was the main activation function for all of the layers with the exception of the last which used the sigmoid activation function.  Binary Crossentropy was utilized as the loss function and the Adaptive Moment Estimation (Adam) variant of stochastic gradient descent was used with a batch size of 32 for 15 epochs.

## Accuracy
This method was able to achieve a 90.70% development accuracy during training and resulted in a 81.25% testing accuracy during testing. Though not as accurate as many of the state-of-the-art systems currently out there, this still is a decent result considering the relatively small dataset size and it paves the way for further refinements for greater accuracy.

## Reproducable Results
A fully trained model is provided as `model.ker` which is the model I trained using the above methods. There is also the `test-yhat` file which is the list of predictions using that model for the testing set. I also provided a method in `train.py` that when uncommented in the main function ensures that all relevant places of randomness are seeded and that TensorFlow is only running on one thread to exactly reproduce my results. Note that because doing this requires TensorFlow to only run on one thread, this results in a massive slowdown and is not recommened if you just want to use your own results.

## License
With the sole exception of the files located under the `dataset/train/` directory and the files located under the `testset/test/` directory, all files of this software and associated documentation files are licensed under the Apache License, Version 2.0.

The files located under the `dataset/train/` directory and the files located under the `testset/test/` directory, and only the files located under the `dataset/train/` directory and the files located under the `testset/test/` directory, are in the public domain.

Copyright 2018 Dan Tran

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
