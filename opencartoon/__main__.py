"""The Main of the Image Cartoonifyer with OpenCV
-----------------------------

About this Module
------------------
This module is the main entry point of The Main of the Image Cartoonifyer
with OpenCV.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-03-30"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import easygui  # to open the filebox


def cartoonify(ImagePath):
    """All the operation will be done on the button click,

    Thus, all the below steps are the part of this function

    :param ImagePath: the path of the image to convert
    """
    pass

def upload():
    """Fileopenbox opens the box to choose file

    Also help us store file path as string. The code opens the file
    box, i.e the pop-up box to choose the file from the device, which opens
    every time you run the code. fileopenbox() is the method in easyGUI
    module which returns the path of the chosen file as a string.
    """
    ImagePath = easygui.fileopenbox()
    cartoonify(ImagePath)


if __name__ == '__main__':
    """Main entry point of opencartoon"""
    pass
