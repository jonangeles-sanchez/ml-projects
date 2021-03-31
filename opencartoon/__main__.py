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
import tkinter as tk
import tkinter.messagebox
from pathlib import Path

import cv2
import easygui
import matplotlib.pyplot as plt


def cartoonify():
    """Cartoonify an image

    First, ask for a file path string through a file box and pass it to the
    cartoonify process. This operation will be done on the button click and
    apply a Cartoon effect which has two specialties:
    - Extract Edges from image
    - Highlight Edges and Smooth colors

    Thus, all the below steps are the part of this function. This code will
    convert the image into a numpy array. This then helps to perform operations
    in which cell values depict R, G, and B values of a pixel.
    """
    # Import an image
    image_path = easygui.fileopenbox()
    images = [read_image(image_path)]

    # Transform the image in cartoon
    images += extract_edges(images[0])
    images += highlight_edges(images[0], images[-1])

    # Plot the steps of the transitions
    resized_images = [resize_image(image) for image in images]
    plot_transition(resized_images)
    plt.show()

    enable_save(images[-1], image_path)


def extract_edges(image):
    """Extract edges of an image

    In 3 steps:
    - Converts an image to grayscale
    - Apply median blur to smoothen an image.
        Here, the center pixel is assigned a mean value of all the pixels which
        fall under the kernel.
    - Retrieve the edges for cartoon effect by using thresholding technique.
        Here, we to retrieve the edges and highlight them. This is attained by
        the adaptive thresholding technique. The threshold value is the mean of
        the neighborhood pixel values area minus the constant C. C is a constant
        that is subtracted from the mean or weighted sum of the neighborhood
        pixels. Thresh_binary is the type of threshold applied, and the
        remaining parameters determine the block size.

    :param image: the image to extract edges
    :return: a new image with the edges extracted
    """
    results = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)]
    results += [cv2.medianBlur(results[-1], 5)]
    results += [
        cv2.adaptiveThreshold(
            results[-1], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            9, 9
        )
    ]
    return results


def enable_save(image, image_path):
    """Create a save button for the generated image and displays it

    :param image: the image to save
    :param image_path: the path where to save the image
    """
    save1 = tk.Button(top, text="Save cartoon image",
                      command=lambda: save(image, Path(image_path)), padx=30,
                      pady=5)
    save1.configure(background='#364156', foreground='white',
                    font=('calibri', 10, 'bold'))
    save1.pack(side=tk.TOP, pady=50)


def plot_transition(images):
    """Plots the whole transition

    To plot all the images, we first make a list of all the images. The
    list here is named “images” and contains all the resized images. Now,
    we create axes like subplots in a plot and display one-one images in
    each block on the axis using imshow() method.

    :param images: a list of images to plot
    """
    fig, axes = plt.subplots(3, 2, figsize=(8, 8),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')


def highlight_edges(image, edges):
    """Prepares a lightened color image that we mask with edges

    At the end it produces the cartoon image. We use bilateralFilter which
    removes the noise. It can be taken as smoothening of an image to an extent.

    The third parameter is the diameter of the pixel neighborhood, i.e,
    the number of pixels around a certain pixel which will determine its
    value. The fourth and Fifth parameter defines sigma color and
    sigma space. These parameters are used to give a sigma effect, i.e make
    an image look vicious and like water paint, removing the roughness in
    colors.

    :param image: the image to filter
    :param edges: the edges of the image
    :return: the filtered image
    """
    # Smoothen Image
    results = [cv2.bilateralFilter(image, 9, 300, 300)]

    # Combines smooth image and edges
    results += [cv2.bitwise_and(results[0], results[0], mask=edges)]
    return results


def resize_image(image):
    """Resizes image submitted at a fixed scale of 960x540

    After each transformation, we resize the resultant image using the
    resize method in cv2 and display it using imshow method. This is
    done to get more clear insights into every single transformation step.

    :param image: the image to resize with opencv
    :return: the image resized to 960x540
    """
    resized = cv2.resize(image, (960, 540))
    return resized


def read_image(image_path):
    """Read the image

    The image is stored in form of numbers, confirm that image is chosen

    :param image_path: the path of the image
    :return: the image loadded in numpy array
    """
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    if original_image is None:
        print("Can not find any image. Choose appropriate file")
        sys.exit()
    return original_image


def save(image, image_path):
    """Save an image using imwrite

    For this, the tail of the old path and store the cartoonified image in
    the same folder. Then imwrite save the file at the path mentioned and
    cv2.cvtColor assure that no color get extracted or highlighted while we
    save our image. Thus, at last, the user is given confirmation that the
    image is saved with the name and path of the file.
    
    :param image: the image to save
    :param image_path: the original image path
    """
    new_name = "cartoonified_image"
    path = Path(image_path.parent, new_name + image_path.suffix)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    image = f"Image saved to {path}"
    tk.messagebox.showinfo(title=None, message=image)


if __name__ == '__main__':
    """Main entry point of opencartoon"""
    top = tk.Tk()
    top.geometry('400x400')
    top.title('Labesoft - Cartoonify Your Image!')
    top.configure(background='white')
    label = tk.Label(top, background='#CDCDCD', font=('calibri', 20, 'bold'))
    upload = tk.Button(top, text="Cartoonify an Image", command=cartoonify,
                       padx=10, pady=5)
    upload.configure(background='#364156', foreground='white',
                     font=('calibri', 10, 'bold'))
    upload.pack(side=tk.TOP, pady=50)
    top.mainloop()
