"""The GUI of The Handwritten Recognition Application
-----------------------------

About this Module
------------------
The goal of this module is create a tkinter GUI that recognize handwritten
digits.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-04-27"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import tkinter as tk
from tkinter import *

from PIL import ImageGrab

from digitrecog.__main__ import DigitRecogModel, MODEL_FILENAME


class RecogGUI(tk.Tk):
    """A tkinter GUI to test the handwritten digit recognition capability"""

    def __init__(self, model):
        """Load the tkinter window and initialize it

        :param model: the model loaded
        """
        tk.Tk.__init__(self)
        self.model = model
        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white",
                                cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognize",
                                      command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear",
                                      command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)
        # self.bind("<Motion>", self.motion)

    @staticmethod
    def motion(event):
        """Utility method to track mouse movement"""
        x, y = event.x, event.y
        print('(x={}, y={})'.format(x, y))

    def clear_all(self):
        """Clear the white board removing previous digit"""
        self.canvas.delete("all")

    def classify_handwriting(self):
        """Classify the character written by user"""
        a = self.canvas.winfo_rootx()
        b = self.canvas.winfo_rooty()
        c = a + self.canvas.winfo_width()
        d = b + self.canvas.winfo_height()
        rect = a + 4, b + 4, c - 4, d - 4
        im = ImageGrab.grab(rect)
        digit, acc = self.model.predict(im)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def draw_lines(self, event):
        """Draw line using filled ovals"""
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r,
                                fill='black', outline="")


if __name__ == '__main__':
    """Main entry point of the digitrecog gui"""
    m = DigitRecogModel(MODEL_FILENAME)
    app = RecogGUI(m)
    mainloop()
