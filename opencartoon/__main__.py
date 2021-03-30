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

import os
import sys
import tkinter as tk
import tkinter.messagebox

import cv2
import easygui  # to get fileopenbox
import matplotlib.pyplot as plt


def cartoonify(image_path):
    """All the operation will be done on the button click,

    Thus, all the below steps are the part of this function. This code will
    first:
    - convert our image into a numpy array. imread is a method in cv2
    which is used to store images in the form of numbers. This helps us to
    perform operations according to our needs. The image is read as a numpy
    array, in which cell values depict R, G, and B values of a pixel.

    :param image_path: the path of the image to convert
    """
    original_image = read_image(image_path)

    resized1 = resize_image(original_image)
    resized2, grayscale_image = convert2gray(original_image)
    resized3, smooth_grayscale = apply_blur(grayscale_image)
    resized4, edge_image = extract_edges(smooth_grayscale)
    resized5, color_image = remove_noise(original_image)
    resized6 = combine(color_image, edge_image)

    plot_transition(resized1, resized2, resized3, resized4, resized5, resized6)
    enable_save(resized6, image_path)
    plt.show()


def enable_save(resized6, image_path):
    # save button code
    save1 = tk.Button(top, text="Save cartoon image",
                      command=lambda: save(resized6, image_path), padx=30,
                      pady=5)
    save1.configure(background='#364156', foreground='white',
                    font=('calibri', 10, 'bold'))
    save1.pack(side=tk.TOP, pady=50)


def plot_transition(resized1, resized2, resized3, resized4, resized5, resized6):
    # Plotting the whole transition
    # To plot all the images, we first make a list of all the images. The
    # list here is named “images” and contains all the resized images. Now,
    # we create axes like subplots in a plot and display one-one images in
    # each block on the axis using imshow() method.
    #
    # plt.show() plots the whole plot at once after we plot on each subplot.
    images = [resized1, resized2, resized3, resized4, resized5, resized6]
    fig, axes = plt.subplots(3, 2, figsize=(8, 8),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')


def combine(color_image, edge_image):
    # masking edged image with our "BEAUTIFY" image
    #
    # So, let’s combine the two specialties. This will be done using MASKING.
    # We perform bitwise and on two images to mask them. Remember, images are
    # just numbers?
    #
    # Yes, so that’s how we mask edged image on our “BEAUTIFY” image.
    #
    # This finally CARTOONIFY our image!
    cartoon_image = cv2.bitwise_and(color_image, color_image, mask=edge_image)
    resized6 = resize_image(cartoon_image)
    return resized6


def remove_noise(original_image):
    # applying bilateral filter to remove noise and keep edge sharp as required
    #
    # We finally work on the second specialty. We prepare a lightened color
    # image that we mask with edges at the end to produce a cartoon image. We
    # use bilateralFilter which removes the noise. It can be taken as
    # smoothening of an image to an extent.
    #
    # The third parameter is the diameter of the pixel neighborhood, i.e,
    # the number of pixels around a certain pixel which will determine its
    # value. The fourth and Fifth parameter defines sigma color and
    # sigma space. These parameters are used to give a sigma effect, i.e make
    # an image look vicious and like water paint, removing the roughness in
    # colors.
    #
    # Yes, it’s similar to BEAUTIFY or AI effect in cameras of modern mobile
    # phones.
    color_image = cv2.bilateralFilter(original_image, 9, 300, 300)
    resized5 = resize_image(color_image)
    return resized5, color_image


def extract_edges(smooth_grayscale):
    # retrieving the edges for cartoon effect by using thresholding technique
    #
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
    edge_image = cv2.adaptiveThreshold(smooth_grayscale, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 9, 9)
    resized4 = resize_image(edge_image)
    return resized4, edge_image


def apply_blur(grayscale_image):
    # applying median blur to smoothen an image
    #
    # To smoothen an image, we simply apply a blur effect. This is done using
    # medianBlur() function. Here, the center pixel is assigned a mean value
    # of all the pixels which fall under the kernel. In turn, creating a blur
    # effect.
    smooth_grayscale = cv2.medianBlur(grayscale_image, 5)
    resized3 = resize_image(smooth_grayscale)
    return resized3, smooth_grayscale


def convert2gray(original_image):
    # converting an image to grayscale
    # vtColor(image, flag) is a method in cv2 which is used to transform
    # an image into the colour-space mentioned as ‘flag’. Here, our first
    # step is to convert the image into grayscale. Thus, we use the BGR2GRAY
    # flag. This returns the image in grayscale. A grayscale image is stored
    # as grayscale_image.
    #
    # After each transformation, we resize the resultant image using the
    # resize method in cv2 and display it using imshow method. This is
    # done to get more clear insights into every single transformation step.
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    resized2 = resize_image(grayscale_image)
    return resized2, grayscale_image


def resize_image(original_image):
    resized1 = cv2.resize(original_image, (960, 540))
    return resized1


def read_image(image_path):
    # read the image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # image is stored in form of numbers

    # confirm that image is chosen
    if original_image is None:
        print("Can not find any image. Choose appropriate file")
        sys.exit()
    return original_image


def save(resized6, image_path):
    # saving an image using imwrite
    # Here, the idea is to save the resultant image. For this, we take the
    # old path, and just change the tail (name of the old file) to a new name
    # and store the cartoonified image with a new name in the same folder by
    # appending the new name to the head part of the file.
    #
    # For this, we extract the head part of the file path by os.path.dirname(
    # ) method. Similarly, os.path.splitext(ImagePath)[1] is used to extract
    # the extension of the file from the path.
    #
    # Here, new_name stores “Cartoonified_Image” as the name of a new file.
    # os.path.join(path1, new_name + extension) joins the head of path to the
    # new name and extension. This forms the complete path for the new file.
    #
    # imwrite() method of cv2 is used to save the file at the path mentioned.
    # cv2.cvtColor(resized6, cv2.COLOR_RGB2BGR) is used to assure that no
    # color get extracted or highlighted while we save our image. Thus,
    # at last, the user is given confirmation that the image is saved with
    # the name and path of the file.
    new_name = "cartoonified_Image"
    path1 = os.path.dirname(image_path)
    extension = os.path.splitext(image_path)[1]
    path = os.path.join(path1, new_name + extension)
    cv2.imwrite(path, cv2.cvtColor(resized6, cv2.COLOR_RGB2BGR))
    image = "Image saved by name " + new_name + " at " + path
    tk.messagebox.showinfo(title=None, message=image)


def upload():
    """Ask for a file path and pass it to the cartoonify process

    Also help us store file path as string. The code opens the file
    box, i.e the pop-up box to choose the file from the device, which opens
    every time you run the code. fileopenbox() is the method in easyGUI
    module which returns the path of the chosen file as a string.
    """
    image_path = easygui.fileopenbox()
    cartoonify(image_path)


if __name__ == '__main__':
    """Main entry point of opencartoon"""
    top = tk.Tk()
    top.geometry('400x400')
    top.title('Cartoonify Your Image !')
    top.configure(background='white')
    label = tk.Label(top, background='#CDCDCD', font=('calibri', 20, 'bold'))
    upload = tk.Button(top, text="Cartoonify an Image", command=upload, padx=10,
                       pady=5)
    upload.configure(background='#364156', foreground='white',
                     font=('calibri', 10, 'bold'))
    upload.pack(side=tk.TOP, pady=50)
    top.mainloop()
