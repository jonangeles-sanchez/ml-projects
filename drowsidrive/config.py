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

from datatools.dataconfig import DataConfig


class DrowsiDriveConfig(DataConfig):
    """Config class for the paths of the applciation"""
    # Data path
    SOURCE_PATH = Path("data", "orig")
    TARGET_PATH = Path("data", "target")
    TRAIN_PATH = TARGET_PATH.joinpath("training")
    VAL_PATH = TARGET_PATH.joinpath("validation")
    TEST_PATH = TARGET_PATH.joinpath("testing")

    # Data prepare
    BATCH_SIZE = 32
    TARGET_SIZE = (24, 24)
    INPUT_SHAPE = TARGET_SIZE + (1,)
    TRAIN_SPLIT = 1.0
    VAL_SPLIT = 0.1

    # Model
    MODEL_FILENAME = 'labesoft_cnn_cat2.h5'
    MODEL_PATH = Path('models', MODEL_FILENAME)

    # View Detection
    EYE_OPEN = -1
    NO_FACE = 0
    BOTH_EYES_CLOSED = 1
    LABELS = {
        EYE_OPEN: 'Open',
        NO_FACE: 'No Face Detected',
        BOTH_EYES_CLOSED: 'Close'
    }
    CLASS_PATH = Path('haar cascade files')
    FACE_CCLASS_PATH = CLASS_PATH.joinpath('haarcascade_frontalface_alt.xml')
    RIGHT_EYE_CCLASS_PATH = CLASS_PATH.joinpath(
        'haarcascade_righteye_2splits.xml'
    )
    LEFT_EYE_CCLASS_PATH = CLASS_PATH.joinpath(
        'haarcascade_lefteye_2splits.xml'
    )
