"""The main of The Driver Drowsiness Detection System
-----------------------------

About this Module
------------------
The goal of this module is to run the system using the core functionalities.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-06-04"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

from drowsidrive.model import DrowsiDrive
from drowsidrive.view import DrowsiDriveView


if __name__ == '__main__':
    """Main entry point of the drowsidrive package"""
    # train()
    drowsy_drive = DrowsiDrive()
    view = DrowsiDriveView(drowsy_drive)
    view.main_loop()
