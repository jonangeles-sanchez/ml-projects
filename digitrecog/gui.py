"""The GUI of The Handwritten Recognition Application
-----------------------------

About this Module
------------------
The goal of this module is create a tkinter GUI that recognize the handwritten
charaters.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-04-27"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import tkinter as tk
from tkinter import *

import numpy as np
from PIL import ImageGrab, ImageOps
from keras.models import load_model

# model = load_model('mnist10.h5')
# model = load_model('mnist15.h5')
model = load_model('mnist20.h5')


def predict_digit(img):
    img = ImageOps.invert(img)
    # img.show("Test")
    # resize image to 28x28 pixels
    img = img.resize((28, 28))
    # convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    img = img.astype('float32')
    # reshaping to support our model input and normalizing
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    # predicting the class
    res = model.predict([img])
    return np.argmax(res[0]), max(res[0])


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white",
                                cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognise",
                                      command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear",
                                      command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)
        # self.bind('<Motion>', self.motion)

    @staticmethod
    def motion(event):
        x, y = event.x, event.y
        print('{}, {}'.format(x, y))

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        a = self.canvas.winfo_rootx()
        b = self.canvas.winfo_rooty()
        c = a + self.canvas.winfo_width()
        d = b + self.canvas.winfo_height()
        rect = a + 4, b + 4, c - 4, d - 4
        im = ImageGrab.grab(rect)
        digit, acc = predict_digit(im)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r,
                                fill='black', outline="")


app = App()
mainloop()
