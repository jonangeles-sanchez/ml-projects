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

from pathlib import Path

from datatools.dataconfig import DataConfig


class CancerDetectConfig(DataConfig):
    INPUT_DATASET = Path("data", "orig")

    BASE_PATH = Path("data", "idc")
    TRAIN_PATH = BASE_PATH.joinpath("training")
    VAL_PATH = BASE_PATH.joinpath("validation")
    TEST_PATH = BASE_PATH.joinpath("testing")

    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
