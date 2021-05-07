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

import tensorflow as tf

from keras import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import mnist
from keras.layers import (
    Dense, Flatten, Conv2D
)
from keras.layers import MaxPooling2D


def create_lenet5(shape):
    """
    Implementation of a modified LeNet-5.
    Modified Architecture -- ConvNet --> Pool --> ConvNet --> Pool --> (
    Flatten) --> FullyConnected --> FullyConnected --> Softmax

    Arguments:
    :param shape: -- shape of the images of the dataset

    :return: results -- a Model() instance in Keras
    """
    result = Sequential()

    # Layer 1
    result.add(Conv2D(filters=6, kernel_size=5, strides=1, activation='relu',
                      input_shape=shape, name='convolution_1'))
    result.add(MaxPooling2D(pool_size=2, strides=2, name='max_pool_1'))

    # Layer 2
    result.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='relu',
                      name='convolution_2'))
    result.add(MaxPooling2D(pool_size=2, strides=2, name='max_pool_2'))

    # Layer 3
    result.add(Flatten(name='flatten'))
    result.add(Dense(units=120, activation='relu', name='fully_connected_1'))

    # Layer 4
    result.add(Dense(units=84, activation='relu', name='fully_connected_2'))

    # Output
    result.add(Dense(units=10, activation='softmax', name='output'))
    result._name = 'lenet5'
    return result


if __name__ == '__main__':
    """Main entry point of writerecog"""
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape)

    # Preprocessing
    input_shape = (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], *input_shape)
    x_test = x_test.reshape(x_test.shape[0], *input_shape)

    batch_size = 64
    num_classes = 10
    epochs = 20

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = create_lenet5(input_shape)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # Training
    variable_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                               patience=2)
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                     verbose=1, validation_data=(x_test, y_test),
                     callbacks=[variable_learning_rate])

    print("The model has successfully trained")
    model.save(f"mnist{epochs}.h5")
    print(f"Saving the model as mnist{epochs}.h5")

    # Evaluate
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
