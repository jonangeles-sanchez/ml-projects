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
import random
import shutil
from pathlib import Path

import pandas as pd
from imutils import paths

import config


def split_image_to_dir():
    """This split dataset through copying image into folder

    This is very ineffective.
    """
    original_paths = list(paths.list_images(config.INPUT_DATASET))
    random.seed(7)
    random.shuffle(original_paths)

    index = int(len(original_paths) * config.TRAIN_SPLIT)
    train_paths = original_paths[:index]
    test_paths = original_paths[index:]

    index = int(len(train_paths) * config.VAL_SPLIT)
    val_paths = train_paths[:index]
    train_paths = train_paths[index:]

    datasets = [
        ("training", train_paths, config.TRAIN_PATH),
        ("validation", val_paths, config.VAL_PATH),
        ("testing", test_paths, config.TEST_PATH)
    ]

    for set_type, orig_paths, base_path in datasets:
        print(f'Building {set_type} set')

        if not base_path.exists():
            print(f'Building directory {base_path}')
            os.makedirs(base_path)

        for path in orig_paths:
            file = path.split(os.path.sep)[-1]
            label = file[-5:-4]

            label_path = base_path.joinpath(label)
            if not label_path.exists():
                print(f'Building directory {label_path}')
                os.makedirs(label_path)

            new_path = label_path.joinpath(file)
            shutil.copy2(path, new_path)


def split_df_to_csv():
    """Splits image path to dataframes then exports the content to csv

    This creates a test set, a validation set and a train set, it also returns
    the dataframes created.

    :return: a tuple of the train, validation and test dataframe
    """
    config.BASE_PATH.mkdir(exist_ok=True)
    original_paths = pd.DataFrame(paths.list_images(config.INPUT_DATASET),
                                  columns=['images'])
    train = original_paths.sample(frac=config.TRAIN_SPLIT, random_state=7)

    # Test set
    test = original_paths.drop(train.index)
    test.to_csv((Path(config.BASE_PATH, 'test.csv')))

    # Valid set
    valid = train.sample(frac=config.VAL_SPLIT, random_state=7)
    valid.to_csv(Path(config.BASE_PATH, 'valid.csv'))

    # Train set
    train = train.drop(valid.index)
    train.to_csv((Path(config.BASE_PATH, 'train.csv')))

    return train, valid, test
