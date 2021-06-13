"""The paths configuration of ML project data
-----------------------------

Project structure
-----------------
*datatools/*
    **dataconfig.py**:
        The paths configuration of ML project data

About this Module
------------------
The goal of this module is to regroup all path configuration of the data for ML
projects.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-06-04"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

from pathlib import Path


class Config:
    def __init__(self):
        self.SOURCE_PATH = Path("data", "orig")
        self.TARGET_PATH = Path("data", "target")
