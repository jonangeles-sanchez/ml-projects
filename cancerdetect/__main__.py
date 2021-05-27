"""The main of Breast Cancer Classification with Deep Learning
-----------------------------

About this Module
------------------
This module is the main entry point of The main of Breast Cancer Classification
with Deep Learning.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-05-27"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.np_utils import to_categorical

from cancerdetect import model, config

matplotlib.use("Agg")
NUM_EPOCHS = 40
INIT_LR = 1e-2
BATCH_SIZE = 32


def train():
    len_train, train_df = read('train.csv')
    len_val, val_df = read('valid.csv')
    len_test, test_df = read('test.csv')

    train_df['labels'] = extract_labels(train_df)
    val_df['labels'] = extract_labels(val_df)
    test_df['labels'] = extract_labels(test_df)

    train_labels = to_categorical(train_df['labels'])
    class_totals = train_labels.sum(axis=0)
    class_weight = class_totals.max() / class_totals
    class_weight = dict(enumerate(class_weight.flatten()))

    train_aug = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    )
    train_gen = train_aug.flow_from_dataframe(
        train_df,
        x_col='images',
        y_col='labels',
        target_size=(48, 48),
        color_mode="rgb",
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    val_aug = ImageDataGenerator(rescale=1 / 255.0)
    val_gen = val_aug.flow_from_dataframe(
        val_df,
        x_col='images',
        y_col='labels',
        target_size=(48, 48),
        color_mode="rgb",
        shuffle=False,
        batch_size=BATCH_SIZE
    )

    test_gen = val_aug.flow_from_dataframe(
        test_df,
        x_col='images',
        y_col='labels',
        class_mode="categorical",
        target_size=(48, 48),
        color_mode="rgb",
        shuffle=False,
        batch_size=BATCH_SIZE
    )

    cancer_model = model.build(width=48, height=48, depth=3, classes=2)
    opt = Adagrad(learning_rate=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
    cancer_model.compile(loss="binary_crossentropy", optimizer=opt,
                         metrics=["accuracy"])

    m = cancer_model.fit(
        train_gen,
        steps_per_epoch=len_train // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=len_val // BATCH_SIZE,
        class_weight=class_weight,
        epochs=NUM_EPOCHS
    )

    print("Now evaluating the model")
    test_gen.reset()
    pred_indices = cancer_model.predict(
        test_gen, steps=(len_test // BATCH_SIZE) + 1)

    pred_indices = np.argmax(pred_indices, axis=1)

    print(classification_report(test_gen.classes, pred_indices,
                                target_names=test_gen.class_indices.keys()))

    cm = confusion_matrix(test_gen.classes, pred_indices)
    total = sum(sum(cm))
    accuracy = (cm[0, 0] + cm[1, 1]) / total
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print(cm)
    print(f'Accuracy: {accuracy}')
    print(f'Specificity: {specificity}')
    print(f'Sensitivity: {sensitivity}')

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, NUM_EPOCHS), m.history["loss"], label="train_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), m.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), m.history["acc"], label="train_acc")
    plt.plot(np.arange(0, NUM_EPOCHS), m.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on the IDC Dataset")
    plt.xlabel("Epoch No.")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('plot.png')


def extract_labels(train_df):
    return train_df['images'].apply(lambda x: str(x.split(os.path.sep)[-2]))


def read(csv):
    df = pd.read_csv(Path(config.BASE_PATH, csv))
    return len(df), df


if __name__ == '__main__':
    """Main entry point of cancerdetect"""
    # build_dataset.split()
    train()
