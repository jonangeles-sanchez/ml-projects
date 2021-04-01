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


class OpenCartoon:
    def __init__(self):
        """OpenCartoon first ask for a file path string through a file box

        This path and image will be the essential part of the cartoonify
        process. Thus, this code will convert the image into a numpy array.
        This then helps to perform operations in which cell values depict R, G,
        and B values of a pixel.
        """
        self.image_path = None
        self.images = []

    def cartoonify(self):
        """Transform the image in cartoon

        This operation will be done on the button click and
        apply a Cartoon effect which has two specialties:
         - Extract Edges from image
         - Highlight Edges and Smooth colors
        """
        self.image_path = Path(easygui.fileopenbox())
        self.images = [self.read_image()]

        self.images += self.extract_edges()
        self.images += self.highlight_edges()
        self.plot_transition()
        self.enable_save()

    def enable_save(self):
        """Create a save button for the generated image and displays it"""
        save1 = tk.Button(
            top,
            text="Save cartoon image",
            command=self.save,
            padx=30, pady=5
        )
        save1.configure(
            background='#364156', foreground='white',
            font=('calibri', 10, 'bold')
        )
        save1.pack(side=tk.TOP, pady=50)

    def extract_edges(self):
        """Extract edges of an image

        In 3 steps:
         - Converts an image to grayscale
         - Apply median blur to smoothen an image. Here, the center pixel is
            assigned a mean value of all the pixels which fall under the kernel.
        - Retrieve the edges for cartoon effect by using thresholding technique.
            This retrieves the edges and highlight them. This is obtained
            by the adaptive thresholding technique. The threshold value is the
            mean of the neighborhood pixel values area minus the constant C. C
            is a constant that is subtracted from the mean or weighted sum of
            the neighborhood pixels. THRESH_BINARY is the type of threshold
            applied, and the remaining parameters determine the block size.

        :return: a new image with the edges extracted
        """
        results = [cv2.cvtColor(self.images[0], cv2.COLOR_BGR2GRAY)]
        results += [cv2.medianBlur(results[-1], 5)]
        results += [
            cv2.adaptiveThreshold(
                results[-1], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                9, 9
            )
        ]
        return results

    def highlight_edges(self):
        """Prepares a lightened color image that we mask with edges

        At the end it produces the cartoon image. We use bilateralFilter which
        removes the noise. It can be taken as smoothening of an image to an
        extent.

        The third parameter is the diameter of the pixel neighborhood, i.e,
        the number of pixels around a certain pixel which will determine its
        value. The fourth and Fifth parameter defines sigma color and
        sigma space. These parameters are used to give a sigma effect, i.e make
        an image look vicious and like water paint, removing the roughness in
        colors.

        :return: the filtered image
        """
        # Smoothen the original image
        results = [cv2.bilateralFilter(self.images[0], 9, 300, 300)]

        # Combines smooth image and edges
        results += [
            cv2.bitwise_and(results[0], results[0], mask=self.images[-1])
        ]
        return results

    def plot_transition(self):
        """Plots the whole transition

        To plot all the images, we first make a list of all the images. The
        list here is named “images” and contains all the resized images. Now,
        we create axes like subplots in a plot and display one-one images in
        each block on the axis using imshow() method.
        """
        resized_images = self.resize_images()
        fig, axes = plt.subplots(3, 2, figsize=(8, 8),
                                 subplot_kw={'xticks': [], 'yticks': []},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))
        for i, ax in enumerate(axes.flat):
            ax.imshow(resized_images[i], cmap='gray')
        plt.show()

    def read_image(self):
        """Read the image

        The image is stored in form of numbers, confirm that image is chosen

        :return: the image loadded in numpy array
        """
        original_image = cv2.imread(str(self.image_path))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        if original_image is None:
            print("Can not find any image. Choose appropriate file")
            sys.exit()
        return original_image

    def resize_images(self):
        """Resizes image submitted at a fixed scale of 960x540

        After each transformation, we resize the resultant image using the
        resize method in cv2 and display it using imshow method. This is
        done to get more clear insights into every single transformation step.

        :return: the image resized to 960x540
        """
        resized_images = [
            cv2.resize(image, (960, 540)) for image in self.images
        ]
        return resized_images

    def save(self):
        """Save an image using imwrite

        For this, the tail of the old path and store the cartoonified image in
        the same folder. Then imwrite save the file at the path mentioned and
        cv2.cvtColor assure that no color get extracted or highlighted while we
        save our image. Thus, at last, the user is given confirmation that the
        image is saved with the name and path of the file.
        """
        new_name = "cartoonified_image"
        path = Path(self.image_path.parent, new_name + self.image_path.suffix)
        cv2.imwrite(str(path), cv2.cvtColor(self.images[-1], cv2.COLOR_RGB2BGR))
        tk.messagebox.showinfo(title=None, message=f"Image saved to {path}")


if __name__ == '__main__':
    """Main entry point of opencartoon"""
    top = tk.Tk()
    oc = OpenCartoon()

    top.geometry('400x400')
    top.title('Labesoft - Cartoonify Your Image!')
    top.configure(background='white')
    label = tk.Label(top, background='#CDCDCD', font=('calibri', 20, 'bold'))
    upload = tk.Button(top, text="Cartoonify an Image", command=oc.cartoonify,
                       padx=10, pady=5)
    upload.configure(background='#364156', foreground='white',
                     font=('calibri', 10, 'bold'))
    upload.pack(side=tk.TOP, pady=50)
    top.mainloop()
