"""The main of the Sign Gesture Language Recognition Application
-----------------------------

About this Module
------------------
This module is the main entry point of The main of the Sign Gesture Language
Recognition Application.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-05-12"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import warnings
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, MaxPool2D
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.simplefilter(action='ignore', category=FutureWarning)

# Creating the dimensions for the ROI...
ROI_top = 30
ROI_bottom = 230
ROI_right = 350
ROI_left = 550

word_dict = ['One', 'Ten', 'Two', 'Three', 'Four', 'Five', 'Six',
             'Seven', 'Eight', 'Nine']


def cal_accum_avg(window_frame, bg, acc_weight):
    if bg is None:
        return window_frame.copy().astype("float")
    else:
        cv2.accumulateWeighted(window_frame, bg, acc_weight)
        return bg


def segment_hand(window_frame, bg, threshold=25):
    diff = cv2.absdiff(bg.astype("uint8"), window_frame)
    threshd = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    # Grab the external contours for the image
    contours, hierarchy = cv2.findContours(threshd.copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        return threshd, hand_segment_max_cont


def create_dataset(element, dataset):
    background = None
    accumulated_weight = 0.5
    cam = cv2.VideoCapture(0)
    num_frames = 0
    num_imgs_taken = 0

    while True:
        ret, frame = cam.read()
        # flipping the frame to prevent inverted image of captured frame...
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()
        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
        if num_frames < 60:
            background = cal_accum_avg(
                gray_frame, background, accumulated_weight
            )
            cv2.putText(
                frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
            )
        # Time to configure the hand specifically into the ROI...
        elif num_frames <= 300:
            hand = segment_hand(gray_frame, background)
            cv2.putText(
                frame_copy, f"Adjust hand gesture for nb {element}",
                (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
            # Checking if the hand is actually detected by counting the
            # number of contours detected...
            if hand:
                thresholded, hand_segment = hand
                # Draw contours around hand segment
                cv2.drawContours(
                    frame_copy, [hand_segment + (ROI_right, ROI_top)], -1,
                    (255, 0, 0), 1
                )
                cv2.putText(
                    frame_copy, f"Frame {num_frames} for nb {element}",
                    (70, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )
                # Also display the thresholded image
                cv2.imshow("Thresholded Hand Image", thresholded)
        else:
            # Segmenting the hand region...
            hand = segment_hand(gray_frame, background)

            # Checking if we are able to detect the hand...
            if hand:
                # unpack the thresholded img and the max_contour...
                thresholded, hand_segment = hand
                # Drawing contours around hand segment
                cv2.drawContours(
                    frame_copy, [hand_segment + (ROI_right, ROI_top)], -1,
                    (255, 0, 0), 1
                )
                cv2.putText(
                    frame_copy, str(num_frames), (70, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )
                msg = f"{num_imgs_taken} images taken for {element}"
                cv2.putText(
                    frame_copy, msg, (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2
                )

                # Displaying the thresholded image
                cv2.imshow("Thresholded Hand Image", thresholded)
                if num_imgs_taken <= 300:
                    filepath = Path(
                        f"./{dataset}", str(element), f"{num_imgs_taken}.jpg"
                    )
                    cv2.imwrite(str(filepath), thresholded)
                else:
                    break
                num_imgs_taken += 1
            else:
                cv2.putText(
                    frame_copy, 'No hand detected...', (200, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )

        # Drawing ROI on frame copy
        cv2.rectangle(
            frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom),
            (255, 128, 0), 3
        )
        cv2.putText(
            frame_copy, "Labesoft hand sign recognition _ _ _", (10, 20),
            cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1
        )

        # increment the number of frames for tracking
        num_frames += 1
        # Display the frame with segmented hand
        cv2.imshow("Sign Detection", frame_copy)
        # Closing windows with Esc key...
        # (any other key with ord can be used too.)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    # Releasing the camera & destroying all the windows...
    cv2.destroyAllWindows()
    cam.release()


def train():
    train_path = './train'
    test_path = './test'

    train_batches = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
        directory=train_path, target_size=(64, 64), class_mode='categorical',
        batch_size=10, shuffle=True)
    test_batches = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
        directory=test_path, target_size=(64, 64), class_mode='categorical',
        batch_size=10, shuffle=True)

    imgs, labels = next(train_batches)
    # plot_images(imgs)
    # print(imgs.shape)
    # print(np.where(labels == 1)[1])
    model = create_model()
    # print(model.summary())
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1,
                                  min_lr=0.0001)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2,
                               verbose=0, mode='auto')

    model.compile(optimizer=SGD(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1,
                                  min_lr=0.0005)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2,
                               verbose=0, mode='auto')
    history2 = model.fit(train_batches, epochs=10,
                         callbacks=[reduce_lr, early_stop],
                         validation_data=test_batches)

    # For getting next batch of testing imgs...
    imgs, labels = next(test_batches)

    scores = model.evaluate(imgs, labels, verbose=0)
    print(f'{model.metrics_names[0]} of {scores[0]}; '
          f'{model.metrics_names[1]} of {scores[1] * 100}%')

    # Once the model is fitted we save the model using model.save() function.
    model.save('model.h5')

    predictions = model.predict(imgs, verbose=0)
    print_results(imgs, labels, predictions, word_dict)


def print_results(imgs, labels, predictions, word_dict):
    print("predictions on a small set of test data--")
    for i in np.argmax(predictions, axis=1):
        print(word_dict[i], end='   ')
    print("")
    plot_images(imgs)
    print('Actual labels')
    for i in np.where(labels == 1)[1]:
        print(word_dict[i], end='   ')


def create_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                     input_shape=(64, 64, 3)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                     padding='valid'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(128, activation="relu"))
    # model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    # model.add(Dropout(0.3))
    model.add(Dense(10, activation="softmax"))
    return model


# Plotting the images...
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(30, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def predict():
    model = keras.models.load_model("./model.h5")
    background = None
    accumulated_weight = 0.5
    cam = cv2.VideoCapture(0)
    num_frames = 0

    while True:
        ret, frame = cam.read()
        # flipping the frame to prevent inverted image of captured frame
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()
        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

        if num_frames < 60:
            background = cal_accum_avg(gray_frame, background,
                                       accumulated_weight)

            cv2.putText(
                frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            # segmenting the hand region
            hand = segment_hand(gray_frame, background)

            # Checking if we are able to detect the hand...
            if hand:
                thresholded, hand_segment = hand

                # Drawing contours around hand segment
                cv2.drawContours(
                    frame_copy, [hand_segment + (ROI_right, ROI_top)], -1,
                    (255, 0, 0), 1
                )

                cv2.imshow("Thesholded Hand Image", thresholded)

                thresholded = cv2.resize(thresholded, (64, 64))
                thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
                thresholded = np.reshape(
                    thresholded,
                    (1, thresholded.shape[0], thresholded.shape[1], 3)
                )

                pred = model.predict(thresholded)
                cv2.putText(
                    frame_copy, word_dict[np.argmax(pred)], (170, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )

        # Draw ROI on frame_copy
        cv2.rectangle(
            frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom),
            (255, 128, 0), 3
        )

        # incrementing the number of frames for tracking
        num_frames += 1

        # Display the frame with segmented hand
        cv2.putText(
            frame_copy, "Hand sign recognition_ _ _", (10, 20),
            cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1
        )
        cv2.imshow("Sign Detection", frame_copy)

        # Close windows with Esc
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

    # Release the camera and destroy all the windows
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    """Main entry point of signgesture"""
    # create_dataset(1, "train")
    # train()
    predict()
