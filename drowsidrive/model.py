"""The model of the Driver Drowsiness Detection System
-----------------------------

About this Module
------------------
The first goal of this module is to create a model with the eyes open/close data
set. After the model was created it can be use by the DrowsiDrive application
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-06-04"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import numpy as np
from pygame import mixer

from tensorflow.keras.layers import (
    Dropout, Conv2D, Flatten, Dense, MaxPooling2D
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.saving.save import load_model

from datatools import dataset
from drowsidrive.config import DrowsiDriveConfig


def prepare_data(config):
    """Prepare the data that is being used for training

    :param config: the paths of the dataset
    :return:
    """
    train_df = dataset.read('train.csv', config)
    valid_df = dataset.read('valid.csv', config)
    gen = image.ImageDataGenerator(rescale=1. / 255)
    return (
        gen.flow_from_dataframe(
            train_df,
            x_col='images', y_col='labels', color_mode='grayscale',
            target_size=config.TARGET_SIZE, batch_size=config.BATCH_SIZE
        ),
        gen.flow_from_dataframe(
            valid_df,
            x_col='images', y_col='labels', color_mode='grayscale',
            target_size=config.TARGET_SIZE, batch_size=config.BATCH_SIZE
        )
    )


def build_model(train_batch, valid_batch, config):
    """Create, train and save the model for the drowsidrive application

    :param config: the config of the paths of the model
    :param train_batch: the training data
    :param valid_batch: the validation data
    :return: the model built
    """
    model = Sequential([
        Conv2D(
            32, kernel_size=(3, 3), activation='relu',
            input_shape=config.INPUT_SHAPE
        ),
        MaxPooling2D(pool_size=(1, 1)),
        # 32 convolution filters used each of size 3x3 again
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(1, 1)),
        # 64 convolution filters used each of size 3x3
        Conv2D(64, (3, 3), activation='relu'),
        # choose the best features via pooling
        MaxPooling2D(pool_size=(1, 1)),
        # randomly turn neurons on and off to improve convergence
        Dropout(0.25),
        # flatten since too many dimensions, classification output only
        Flatten(),
        # fully connected to get all relevant data
        Dense(128, activation='relu'),
        # one more dropout for convergence
        Dropout(0.5),
        # output a softmax to squash the matrix into output probabilities
        Dense(4, activation='softmax')
    ])
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
    )
    model.fit_generator(
        train_batch, validation_data=valid_batch, epochs=15,
        steps_per_epoch=len(train_batch.classes) // config.BATCH_SIZE,
        validation_steps=len(valid_batch.classes) // config.BATCH_SIZE
    )
    model.save(str(config.MODEL_PATH), overwrite=True)
    return model


def train():
    """Train a new model using data and model paths configs

    :return: the built model
    """
    config = DrowsiDriveConfig()
    # dataset.split_df_to_csv(config)

    # Prepare data to train and validate results
    train_data, valid_data = prepare_data(config)
    model = build_model(train_data, valid_data, config)
    return model


class DrowsiDrive:
    """The backend of the drowsiness detection application"""
    def __init__(self):
        """Initialize the alarm params, the model and faces parts classifiers"""
        # Prepare alarm
        mixer.init()
        self.sound = mixer.Sound('alarm.wav')
        self.score = 0
        self.__thickness = 2
        self.config = DrowsiDriveConfig()

        # Prepare model
        self.model = load_model(str(self.config.MODEL_PATH))

        # Classifiers
        self.face_classifier = None
        self.left_eye_classifier = None
        self.right_eye_classifier = None

    @property
    def thickness(self):
        """Get the alert frame thickness

        :return: the alert frame thickness
        """
        return self.__thickness

    @thickness.setter
    def thickness(self, inc):
        """Calculate alert frame thickness using an increment

        :param inc: the increment number
        """
        if self.__thickness < 16:
            self.__thickness += inc
        else:
            self.__thickness -= inc
            if self.__thickness < inc:
                self.__thickness = inc

    def set_classifiers(self, fcc, lcc, rcc):
        """Set all classifiers needed to detect faces and eyes

        :param fcc: the face classifier
        :param lcc: the left eye classifier
        :param rcc: the right eye classifier
        """
        self.face_classifier = fcc
        self.left_eye_classifier = lcc
        self.right_eye_classifier = rcc

    def handle_score(self, drowsiness_state):
        """Calculate score based on drowsiness detected

        :param drowsiness_state: the drowsiness of a face detected in an image
        """
        self.score += drowsiness_state
        if self.score < 0:
            self.score = 0

    def play_alert_sound(self):
        """Play the sound of the alert"""
        try:
            self.sound.play()
        except Exception as err:
            print(err)

    def detect_faces_parts(self, gray):
        """Get the faces of the current gray-scale image

        :param gray: the gray-scale image
        :return: the parts detected (faces, left eyes, right eyes)
        """
        faces = self.face_classifier.detectMultiScale(
            gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25)
        )
        left_eyes = self.left_eye_classifier.detectMultiScale(gray)
        right_eyes = self.right_eye_classifier.detectMultiScale(gray)
        return faces, left_eyes, right_eyes

    def get_state(self, eye):
        """Get the state open/close of the eye provided

        The 24x24 image of an eye is normalized and the state predicted by a
        CNN model previously built and provided by this object

        :param eye: the eye to get open/close state
        :return: the open/close state of the eye
        """
        eye = eye / 255
        eye = eye.reshape(*self.config.TARGET_SIZE, -1)
        eye = np.expand_dims(eye, axis=0)
        return np.argmax(self.model.predict(eye))

    @staticmethod
    def get_slice(image_in, coord):
        """Get the slice of an image

        :param image_in: the image to slice
        :param coord: the coordinates of the slice
        :return: the slice of the image following the coord provided
        """
        x, y, w, h = coord
        return image_in[y:y + h, x:x + w]
