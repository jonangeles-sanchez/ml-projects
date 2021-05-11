"""The Main of the Face Mask Detector Application
-----------------------------

About this Module
------------------
This module is the main entry point of The Main of the Face Mask Detector
Application.

"""

__author__ = "Benoit Lapointe"
__date__ = "2021-05-11"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import os
import sys

import cv2
import numpy as np

from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = "./train"
VALIDATION_DIR = "./valid"
MODEL_FILENAME = "./model-010.h5"


def train():
    model = Sequential([
        Conv2D(100, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(100, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(50, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                        batch_size=10,
                                                        target_size=(150, 150))
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        batch_size=10,
        target_size=(150, 150))
    # checkpoint = ModelCheckpoint('model2-{epoch:03d}.model',
    #                              monitor='val_loss',
    #                              verbose=0, save_best_only=True, mode='auto')
    model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        # callbacks=[checkpoint]
    )
    model.save(MODEL_FILENAME)


def evaluate():
    model = load_model(MODEL_FILENAME)
    results = {0: 'without mask', 1: 'mask'}
    gr_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
    rect_size = 4
    cap = cv2.VideoCapture(0)
    haarcascade = cv2.CascadeClassifier(
        f'{os.path.dirname(sys.executable)}/Lib/site-packages/cv2/data'
        '/haarcascade_frontalface_default.xml'
    )
    while True:
        (rval, im) = cap.read()
        im = cv2.flip(im, 1, 1)

        rerect_size = cv2.resize(im, (
            im.shape[1] // rect_size, im.shape[0] // rect_size))
        faces = haarcascade.detectMultiScale(rerect_size)
        for f in faces:
            (x, y, w, h) = [v * rect_size for v in f]

            face_img = im[y:y + h, x:x + w]
            rerect_sized = cv2.resize(face_img, (150, 150))
            normalized = rerect_sized / 255.0
            reshaped = np.reshape(normalized, (1, 150, 150, 3))
            reshaped = np.vstack([reshaped])
            result = model.predict(reshaped)

            label = np.argmax(result, axis=1)[0]

            cv2.rectangle(im, (x, y), (x + w, y + h), gr_dict[label], 2)
            cv2.rectangle(im, (x, y - 40), (x + w, y), gr_dict[label], -1)
            cv2.putText(im, results[label], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('LIVE', im)
        key = cv2.waitKey(10)

        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    """Main entry point of facemask"""
    # train()
    evaluate()
