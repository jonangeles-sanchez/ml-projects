"""The model of the Breast Cancer Classification with Deep Learning
-----------------------------

About this Module
------------------
The goal of this module is to build a CNN (Convolutional Neural Network).
This network performs the following operations:

    Use 3×3 CONV filters
    Stack these filters on top of each other
    Perform max-pooling
    Use depthwise separable convolution (more efficient, takes up less memory).

We use the Sequential API to build CancerNet and SeparableConv2D to implement
depthwise convolutions. The class CancerNet has a static method build that takes
four parameters- width and height of the image, its depth (the number of color
channels in each image), and the number of classes the network will predict
between, which, for us, is 2 (0 and 1).

In this method, we initialize model and shape. When using channels_first,
we update the shape and the channel dimension.

Now, we’ll define three DEPTHWISE_CONV => RELU => POOL layers; each with a
higher stacking and a greater number of filters. The softmax classifier
outputs prediction percentages for each class. In the end, we return the model.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-05-27"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.backend_config import image_data_format
from tensorflow.python.keras.layers import (
    SeparableConv2D, Activation,
    BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
)


def build(width, height, depth, classes):
    model = Sequential()
    shape = (height, width, depth)
    channel_dim = -1

    if image_data_format() == "channels_first":
        shape = (depth, height, width)
        channel_dim = 1

    model.add(SeparableConv2D(32, (3, 3), padding="same", input_shape=shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channel_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(SeparableConv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channel_dim))
    model.add(SeparableConv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channel_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(SeparableConv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channel_dim))
    model.add(SeparableConv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channel_dim))
    model.add(SeparableConv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channel_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model
