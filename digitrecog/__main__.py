"""The Main module of The Handwritten Digit Recognition Application
-----------------------------

About this Module
------------------
This module is the main entry point of The Main module of The Handwritten
Recognition Application.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-04-27"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import os

import numpy as np
from PIL import ImageOps
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D
)
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import load_model

MODEL_FILENAME = "digit_model.h5"
EPOCH = 16


def create_model(shape):
    """Implementation of a modified LeNet-5.

    Modified Architecture -- ConvNet --> Pool --> ConvNet --> Pool --> (
    Flatten) --> FullyConnected --> FullyConnected --> Softmax

    :param shape: -- shape of the images of the dataset
    :return: result -- a Model() instance in Keras
    """
    result = Sequential()

    # Layer 1
    result.add(
        Conv2D(filters=6, kernel_size=5, strides=1, activation='relu',
               input_shape=shape, name='convolution_1'))
    result.add(MaxPooling2D(pool_size=2, strides=2, name='max_pool_1'))

    # Layer 2
    result.add(
        Conv2D(filters=16, kernel_size=5, strides=1, activation='relu',
               name='convolution_2'))
    result.add(MaxPooling2D(pool_size=2, strides=2, name='max_pool_2'))

    # Layer 3
    result.add(Flatten(name='flatten'))
    result.add(
        Dense(units=120, activation='relu', name='fully_connected_1'))

    # Layer 4
    result.add(Dense(units=84, activation='relu', name='fully_connected_2'))

    # Output
    result.add(Dense(units=10, activation='softmax', name='output'))
    result._name = 'lenet5'
    return result


def train():
    """Train a model from mnist dataset"""
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocessing
    input_shape = (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], *input_shape)
    x_test = x_test.reshape(x_test.shape[0], *input_shape)
    num_classes = 10

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    model = create_model(input_shape)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # Training
    variable_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                               patience=2)
    model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=EPOCH,
        validation_data=(x_test, y_test),
        callbacks=[variable_learning_rate]
    )
    model.save(MODEL_FILENAME)

    # Evaluate
    model.evaluate(x_test, y_test, verbose=0)


class DigitRecogModel:
    """A model that can recognize an alpha character image"""

    def __init__(self, model_fn):
        """Initialize an empty model"""
        if not os.path.isfile(model_fn):
            train()
        self.model = load_model(model_fn)

    def predict(self, img):
        """Prepare the image grabbed and try to recognize the digit

        :param img: the image grabbed by the gui
        :return: the prediction using current model
        """
        img = ImageOps.invert(img)
        # img.show("Test")
        # resize image to 28x28 pixels
        img = img.resize((28, 28))
        # convert rgb to grayscale
        img = img.convert('L')
        img = np.array(img)
        img = img.astype('float32')
        # reshaping to support our model input and normalizing
        img = img.reshape((1, 28, 28, 1))
        img = img / 255.0
        # predicting the class
        res = self.model.predict([img])
        return np.argmax(res[0]), max(res[0])


if __name__ == '__main__':
    """Main entry point of digitrecog"""
    if not os.path.isfile(MODEL_FILENAME):
        train()
    else:
        print(f"File exists {MODEL_FILENAME}")
