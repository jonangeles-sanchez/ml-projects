"""The GUI of The Handwritten Recognition Application
-----------------------------

About this Module
------------------
The goal of this module is create a tkinter GUI that recognize handwritten
characters.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-04-27"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"


from alpharecog.__main__ import AlphaRecogModel, MODEL_FILENAME
from digitrecog.gui import RecogGUI

if __name__ == '__main__':
    """Main entry point of the alpharecog gui"""
    m = AlphaRecogModel(MODEL_FILENAME)
    app = RecogGUI(m)
    app.mainloop()
