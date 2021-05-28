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
from collections import namedtuple
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical

from cancerdetect import model, config

BATCH_SIZE = 32
INIT_LR = 1e-2
IMAGE_SIZE = 48
MODEL_FILENAME = 'model_cancer.h5'
NUM_EPOCHS = 10

Data = namedtuple('Data', ['gen', 'size'])


def train_model():
    """Train the model, plot epochs and evaluate the results"""
    class_weight, train, val = prepare_train_data()
    cancer_model, m = create_model(class_weight, train, val)
    plot(m)
    evaluate(cancer_model, prepare_test_data())


def evaluate(cancer_model, test):
    """Evaluate the cancer detection model

    Based on confusion matrix, accuracy, specificity and sensitivity

    :param cancer_model: the model to evaluate
    :param test: the test image iterator with data size
    """
    print("Now evaluating the model")
    pred_indices = cancer_model.predict(
        test.gen, steps=(test.size // BATCH_SIZE) + 1
    )
    print(
        classification_report(
            test.gen.classes,
            np.argmax(pred_indices, axis=1),
            target_names=test.gen.class_indices.keys()
        )
    )
    cm = confusion_matrix(test.gen.classes, pred_indices)
    print(cm)
    print(f'Accuracy: {(cm[0, 0] + cm[1, 1]) / sum(sum(cm))}')
    print(f'Specificity: {cm[1, 1] / (cm[1, 0] + cm[1, 1])}')
    print(f'Sensitivity: {cm[0, 0] / (cm[0, 0] + cm[0, 1])}')


def plot(m):
    """Plot a history graphic of loss and accuracy across epochs

    :param m: the model training history
    """
    matplotlib.use("Agg")
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, NUM_EPOCHS), m.history["loss"], label="train_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), m.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), m.history["accuracy"], label="train_acc")
    plt.plot(
        np.arange(0, NUM_EPOCHS), m.history["val_accuracy"], label="val_acc"
    )
    plt.title("Training Loss and Accuracy on the IDC Dataset")
    plt.xlabel("Epoch No.")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('plot.png')


def create_model(class_weight, train, val):
    """Build, compile and fit the cancer detection model

    :param class_weight: the weight dict of the 0 and 1 classes
    :param train: the train set image iterator with data size
    :param val: the validation set image iterator with data size
    :return: a tuple of the model and the model training history
    """
    cancer_model = model.build(
        width=IMAGE_SIZE, height=IMAGE_SIZE, depth=3, classes=2
    )
    opt = Adagrad(learning_rate=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
    cancer_model.compile(loss="binary_crossentropy", optimizer=opt,
                         metrics=["accuracy"])
    m = cancer_model.fit(
        train.gen,
        steps_per_epoch=train.size // BATCH_SIZE,
        validation_data=val.gen,
        validation_steps=val.size // BATCH_SIZE,
        class_weight=class_weight,
        epochs=NUM_EPOCHS
    )
    cancer_model.save('model_cancer.h5')
    return cancer_model, m


def prepare_train_data():
    """Prepare all data needed to train a cancer detection model

    :return: a tuple of the class weight, train set and validation set iterators
    """
    # Datasets
    train_df = read('train.csv')
    val_df = read('valid.csv')

    # Weights
    class_totals = to_categorical(train_df['labels']).sum(axis=0)
    class_weight = class_totals.max() / class_totals
    class_weight = dict(enumerate(class_weight.flatten()))

    # Iterators
    train_gen = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    ).flow_from_dataframe(
        train_df,
        x_col='images',
        y_col='labels',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        color_mode="rgb",
        shuffle=True,
        batch_size=BATCH_SIZE
    )
    val_gen = ImageDataGenerator(rescale=1 / 255.0).flow_from_dataframe(
        val_df,
        x_col='images',
        y_col='labels',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        color_mode="rgb",
        shuffle=False,
        batch_size=BATCH_SIZE
    )

    # Return weight and Data Tuples
    return (
        class_weight,
        Data(train_gen, len(train_df)),
        Data(val_gen, len(val_df))
    )


def prepare_test_data():
    """Prepare data to test a model

    :return: the test set iterator
    """
    test_df = read('test.csv')
    test_gen = ImageDataGenerator(rescale=1 / 255.0).flow_from_dataframe(
        test_df,
        x_col='images',
        y_col='labels',
        class_mode="categorical",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        color_mode="rgb",
        shuffle=False,
        batch_size=BATCH_SIZE
    )
    return Data(test_gen, len(test_df))


def extract_labels(train_df):
    """Extract class labels from the training set

    :param train_df: the training set
    :return: a set of the extracted class labels
    """
    return train_df['images'].apply(lambda x: str(x.split(os.path.sep)[-2]))


def read(csv):
    """Creates a dataframe from a csv file

    :param csv: the csv file name
    :return: the dataframe issued from the csv file
    """
    df = pd.read_csv(Path(config.BASE_PATH, csv))
    df['labels'] = extract_labels(df)
    return df


def test_model():
    """Test a breast cancer detection model loaded from file"""
    cancer_model = load_model(MODEL_FILENAME)
    test = prepare_test_data()
    evaluate(cancer_model, test)


if __name__ == '__main__':
    """Main entry point of cancerdetect"""
    # build_dataset.split()
    train_model()
    # test_model()
