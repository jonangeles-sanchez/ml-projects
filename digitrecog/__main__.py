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

from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.layers import MaxPooling2D

if __name__ == '__main__':
    """Main entry point of writerecog"""
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape)

    # Preprocessing
    input_shape = (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], *input_shape)
    x_test = x_test.reshape(x_test.shape[0], *input_shape)

    batch_size = 128
    num_classes = 10
    epochs = 15

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

    # Modeling
    model = tf.keras.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # Training
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                     verbose=1, validation_data=(x_test, y_test))
    print("The model has successfully trained")
    model.save('mnist15.h5')
    print("Saving the model as mnist.h5")

    # Evaluate
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
