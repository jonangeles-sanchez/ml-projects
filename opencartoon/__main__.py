"""The Main of the Image Cartoonifyer with OpenCV
-----------------------------

About this Module
------------------
This module is the main entry point of The Main of the Image Cartoonifyer
with OpenCV. To convert an image to a cartoon, multiple transformations are
done. Firstly, an image is converted to a Grayscale image. Then,
the Grayscale image is smoothened, and we try to extract the edges in the
image. Finally, we form a color image and mask it with edges. This creates a
beautiful cartoon image with edges and lightened color of the original image.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-03-30"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import sys

import cv2
import easygui  # to open the filebox


def cartoonify(ImagePath):
    """All the operation will be done on the button click,

    Thus, all the below steps are the part of this function. This code will
    first convert our image into a numpy array.Imread is a method in cv2
    which is used to store images in the form of numbers. This helps us to
    perform operations according to our needs. The image is read as a numpy
    array, in which cell values depict R, G, and B values of a pixel.

    :param ImagePath: the path of the image to convert
    """
    # read the image
    originalmage = cv2.imread(ImagePath)
    originalmage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2RGB)
    # print(image)  # image is stored in form of numbers

    # confirm that image is chosen
    if originalmage is None:
        print("Can not find any image. Choose appropriate file")
        sys.exit()

    ReSized1 = cv2.resize(originalmage, (960, 540))
    # plt.imshow(ReSized1, cmap='gray')


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
