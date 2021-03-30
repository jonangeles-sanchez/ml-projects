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

    # converting an image to grayscale
    # vtColor(image, flag) is a method in cv2 which is used to transform
    # an image into the colour-space mentioned as ‘flag’. Here, our first
    # step is to convert the image into grayscale. Thus, we use the BGR2GRAY
    # flag. This returns the image in grayscale. A grayscale image is stored
    # as grayScaleImage.
    #
    # After each transformation, we resize the resultant image using the
    # resize() method in cv2 and display it using imshow() method. This is
    # done to get more clear insights into every single transformation step.
    grayScaleImage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2GRAY)
    ReSized2 = cv2.resize(grayScaleImage, (960, 540))
    # plt.imshow(ReSized2, cmap='gray')

    # applying median blur to smoothen an image
    # To smoothen an image, we simply apply a blur effect. This is done using
    # medianBlur() function. Here, the center pixel is assigned a mean value
    # of all the pixels which fall under the kernel. In turn, creating a blur
    # effect.
    smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
    ReSized3 = cv2.resize(smoothGrayScale, (960, 540))
    # plt.imshow(ReSized3, cmap='gray')

    # retrieving the edges for cartoon effect
    # by using thresholding technique
    # Cartoon effect has two specialties:
    #
    # - Highlighted Edges
    # - Smooth colors
    #
    # In this step, we will work on the first specialty. Here, we will try to
    # retrieve the edges and highlight them. This is attained by the adaptive
    # thresholding technique. The threshold value is the mean of the
    # neighborhood pixel values area minus the constant C. C is a constant
    # that is subtracted from the mean or weighted sum of the neighborhood
    # pixels. Thresh_binary is the type of threshold applied, and the
    # remaining parameters determine the block size.
    getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 9, 9)
    ReSized4 = cv2.resize(getEdge, (960, 540))
    # plt.imshow(ReSized4, cmap='gray')


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
