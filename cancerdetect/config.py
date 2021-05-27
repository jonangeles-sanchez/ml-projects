"""The configurations of Breast Cancer Classification with Deep Learning
-----------------------------

About this Module
------------------
The goal of this module is to declare the path to the input dataset (
data/orig), that for the new directory (data/idc), and the paths
for the training, validation, and testing directories using the base path. We
also declare that 80% of the entire dataset will be used for training,
and of that, 10% will be used for validation.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-05-27"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import os

INPUT_DATASET = "data/orig"

BASE_PATH = "data/idc"
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
