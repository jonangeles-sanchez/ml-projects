"""The Main module of The Handwritten Recognition Application
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

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


if __name__ == '__main__':
    """Main entry point of writerecog"""
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape)
