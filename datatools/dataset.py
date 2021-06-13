"""The dataset builder of the Breast Cancer Classification with Deep Learning
-----------------------------

About this Module
------------------
In this, we’ll import from config, imutils, random, shutil, and os. We
build a list of original paths to the images, then shuffle the list. Then,
we calculate an index by multiplying the length of this list by 0.8 so we can
slice this list to get sublists for the training and testing datasets. Next,
we further calculate an index saving 10% of the list for the training dataset
for validation and keeping the rest for training itself.

Now, datasets is a list with tuples for information about the training,
validation, and testing sets. These hold the paths and the base path for
each. For each setType, path, and base path in this list, we’ll print, say,
‘Building testing set’. If the base path does not exist, we’ll create the
directory. And for each path in originalPaths, we’ll extract the filename and
the class label. We’ll build the path to the label directory(0 or 1)- if it
doesn’t exist yet, we’ll explicitly create this directory. Now, we’ll build
the path to the resulting image and copy the image here- where it belongs.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-05-27"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import os
from pathlib import Path

import pandas as pd
from imutils import paths


def split_df_to_csv(config):
    """Splits image path to dataframes then exports the content to csv

    This creates a test set, a validation set and a train set, it also returns
    the dataframes created.

    :return: a tuple of the train, validation and test dataframe
    """
    config.TARGET_PATH.mkdir(exist_ok=True)
    original_paths = pd.DataFrame(
        paths.list_images(config.SOURCE_PATH), columns=['images']
    )
    train = original_paths.sample(frac=config.TRAIN_SPLIT, random_state=7)

    # Test set
    test = original_paths.drop(train.index)
    test.to_csv((Path(config.TARGET_PATH, 'test.csv')))

    # Valid set
    valid = train.sample(frac=config.VAL_SPLIT, random_state=7)
    valid.to_csv(Path(config.TARGET_PATH, 'valid.csv'))

    # Train set
    train = train.drop(valid.index)
    train.to_csv((Path(config.TARGET_PATH, 'train.csv')))

    return train, valid, test


def read(csv_filename, config):
    """Creates a dataframe from a csv file

    :param config: a data config object that contains project paths
    :param csv_filename: the csv file name
    :return: the dataframe issued from the csv file
    """
    df = pd.read_csv(Path(config.TARGET_PATH, csv_filename))
    df['labels'] = extract_labels(df)
    return df


def extract_labels(train_df):
    """Extract class labels from the training set

    :param train_df: the training set
    :return: a set of the extracted class labels
    """
    return train_df['images'].apply(lambda x: str(x.split(os.path.sep)[-2]))
