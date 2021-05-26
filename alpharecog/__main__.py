"""The main of the Alphabet recognition software
-----------------------------

About this Module
------------------
This module is the main entry point of The main of the Alphabet recognition
software.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-05-26"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import os.path
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import load_model

MODEL_FILENAME = 'alpha_model.h5'
WORD_DICT = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def create_model():
    """Create the alphabet recognition model

    :return: the model prepared
    """
    model = Sequential()
    model.add(
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
               input_shape=(28, 28, 1))
    )
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
               padding='same')
    )
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
               padding='valid')
    )
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(26, activation="softmax"))
    return model


def train():
    """Prepare the data, create, train and save the model.

    Also print the summary of the model
    """
    data = pd.read_csv(
        str(Path('data', 'A_Z Handwritten Data.csv'))
    ).astype('float32')

    # Prepare data
    x = data.drop('0', axis=1)
    y = data['0']

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train = np.reshape(
        x_train.values, (x_train.shape[0], 28, 28)
    )
    x_test = np.reshape(
        x_test.values, (x_test.shape[0], 28, 28)
    )

    # Count elements
    df = pd.DataFrame({'y': y})
    result = pd.DataFrame(
        {'y': df.apply(lambda x: WORD_DICT[int(x)], axis=1)}
    )
    result['y'].value_counts().sort_index(ascending=False).plot(kind='barh')

    # Show data overview
    x_train_shuffled = shuffle(x_train[:100])
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    axes = ax.flatten()
    for i in range(9):
        _, shu = cv2.threshold(
            x_train_shuffled[i], 30, 200, cv2.THRESH_BINARY
        )
        axes[i].imshow(
            np.reshape(x_train_shuffled[i], (28, 28)), cmap="Greys"
        )
    plt.show()

    # Reshape tensors
    x_train_r = np.reshape(
        x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    )
    x_test_r = np.reshape(
        x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    )
    y_train_ohe = to_categorical(y_train, num_classes=26, dtype='int')
    y_test_ohe = to_categorical(y_test, num_classes=26, dtype='int')

    # Model
    model = create_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        x_train_r, y_train_ohe,
        epochs=2,
        validation_data=(x_test_r, y_test_ohe)
    )
    model.save(MODEL_FILENAME)
    model.evaluate(x_test_r, y_test_ohe, verbose=0)


class AlphaRecogModel:
    """A model that can recognize an alpha character image"""

    def __init__(self, model_fn):
        """Initialize an empty model"""
        if not os.path.isfile(model_fn):
            train()
        self.model = load_model(model_fn)

    def predict(self, img):
        """Prepare the image grabbed and try to recognize the character

        :param img: the image grabbed by the gui
        :return: the prediction using current model
        """
        img_copy = np.array(img)
        img_copy = cv2.GaussianBlur(img_copy, (7, 7), 0)
        img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
        img_final = cv2.resize(img_thresh, (28, 28))
        img_final = np.reshape(img_final, (1, 28, 28, 1))
        pred = self.model.predict(img_final)
        img_pred = WORD_DICT[np.argmax(pred)]
        return img_pred, max(pred[0])


if __name__ == '__main__':
    """Main entry point of alpharecog"""
    if not os.path.isfile(MODEL_FILENAME):
        train()
    else:
        print(f"File exists {MODEL_FILENAME}")
