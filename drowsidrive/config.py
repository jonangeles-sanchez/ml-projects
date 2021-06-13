"""The path configuration of the Driver Drowsiness Detection System
-----------------------------

About this Module
------------------
The goal of this module is declare the path to the input dataset (
data/orig), that for the new directory (data/target), and the paths
for the training, validation, and testing directories using the base path. We
also declare that 100% of the entire dataset will be used for training,
and of that, 10% will be used for validation.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-06-04"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

from pathlib import Path

from datatools.config import Config


class DrowsiDriveConfig(Config):
    """Config class for the paths of the applciation"""
    def __init__(self):
        super(DrowsiDriveConfig, self).__init__()
        # Data path
        self.TRAIN_PATH = self.TARGET_PATH.joinpath("training")
        self.VAL_PATH = self.TARGET_PATH.joinpath("validation")
        self.TEST_PATH = self.TARGET_PATH.joinpath("testing")

        # Data prepare
        self.BATCH_SIZE = 32
        self.TARGET_SIZE = (24, 24)
        self.INPUT_SHAPE = self.TARGET_SIZE + (1,)
        self.TRAIN_SPLIT = 1.0
        self.VAL_SPLIT = 0.1

        # Model
        self.MODEL_FILENAME = 'labesoft_cnn_cat2.h5'
        self.MODEL_PATH = Path('models', self.MODEL_FILENAME)

        # View Detection
        self.EYE_OPEN = -1
        self.NO_FACE = 0
        self.BOTH_EYES_CLOSED = 1
        self.LABELS = {
            self.EYE_OPEN: 'Open',
            self.NO_FACE: 'No Face Detected',
            self.BOTH_EYES_CLOSED: 'Close'
        }
        self.CLASS_PATH = Path('haar cascade files')
        self.FACE_CCLASS_PATH = self.CLASS_PATH.joinpath(
            'haarcascade_frontalface_alt.xml'
        )
        self.RIGHT_EYE_CCLASS_PATH = self.CLASS_PATH.joinpath(
            'haarcascade_righteye_2splits.xml'
        )
        self.LEFT_EYE_CCLASS_PATH = self.CLASS_PATH.joinpath(
            'haarcascade_lefteye_2splits.xml'
        )
