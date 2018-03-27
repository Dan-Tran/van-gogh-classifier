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
Tests an image classifier model on the testing set and saves the
predictions to a file, outputing the accuracy of the model on the
testing set along with the baseline accuracies. This uses Keras with
a TensorFlow backend to load, evaluate, and predict using the model.
"""

from keras.models import load_model
import numpy as np

from utils import get_test_data, print_baseline


def evaluate_model(model, testx, testy):
    """
    Evaluates the model on the given testing set. Assumes the model
    tracks accuracy as its only additional metric.

    :param model: The model to be evaluated.
    :param testx: The test image data as a numpy array.
    :param testy: The test image labels as a numpy array.
    """

    score = model.evaluate(testx, testy, batch_size=32)
    print('Test Accuracy: ', score[1])
    print_baseline(testy)


def generate_predictions(model, testx, outfile):
    """
    Generates the predictions by the model on the testing set.

    :param model: The model that will make predictions.
    :param testx: The test image data as a numpy array.
    :param outfile: The name of the output file for the predictions.
    """

    probabilities = model.predict(testx)
    predictions = []
    for prob in probabilities:
        if prob < 0.5:
            predictions.append(0)
        else:
            predictions.append(1)

    np.savetxt(outfile, predictions, fmt='%d')


def main():
    """
    Loads the test data and the model, evaluate the
    test accuracy, and generate predictions.
    """

    # Load in test data and model
    testx, testy = get_test_data()
    model = load_model('model.ker')

    # Evaluate test accuracy
    evaluate_model(model, testx, testy)

    # Generate predictions
    generate_predictions(model, testx, 'test-yhat')


if __name__ == '__main__':
    main()
