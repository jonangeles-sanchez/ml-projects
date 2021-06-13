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

from datatools.config import Config


class CancerDetectConfig(Config):
    def __init__(self):
        super(CancerDetectConfig, self).__init__()
        self.TRAIN_PATH = self.TARGET_PATH.joinpath("training")
        self.VALID_PATH = self.TARGET_PATH.joinpath("validation")
        self.TEST_PATH = self.TARGET_PATH.joinpath("testing")

        self.TRAIN_SPLIT = 0.8
        self.VAL_SPLIT = 0.1
