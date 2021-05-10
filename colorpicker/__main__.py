"""The Main of the Color Detection system using Pandas an OpenCV
-----------------------------

About this Module
------------------
The goal of this module is implement the whole system of the color detection
application. This will detect a color in an image on a mouse click event.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-05-10"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import argparse

import cv2
import pandas as pd


def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, xpos, ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)


def getColorName(R, G, B):
    minimum = 10000
    cname = ""
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(
            B - int(csv.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname


if __name__ == '__main__':
    """Main entry point of __main__"""

    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help="Image Path")
    args = vars(ap.parse_args())
    img_path = args['image']

    # Global vars
    clicked = False
    r = g = b = xpos = ypos = 0

    # Reading image with opencv
    im = cv2.imread(img_path)
    img = cv2.resize(im, (int(3456/5), int(4608/5)))

    # Reading csv file with pandas and giving names to each column
    index = ["color", "color_name", "hex", "R", "G", "B"]
    csv = pd.read_csv('colors.csv', names=index, header=None)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_function)

    while True:
        if cv2.getWindowProperty('image', 0) >= 0:
            cv2.imshow("image", img)
        else:
            break
        if clicked:
            # cv2.rectangle(image, startpoint, endpoint, color, thickness) -1
            # thickness fills rectangle entirely
            cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)

            # Creating text string to display ( Color name and RGB values )
            text = getColorName(r, g, b) + ' R=' + str(r) + ' G=' + str(
                g) + ' B=' + str(b)

            # cv2.putText(img,text,start,font(0-7), fontScale, color, thickness,
            # lineType, (optional bottomLeft bool) )
            cv2.putText(img, text, (50, 50), 2, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)
            # For very light colours we will display text in black colour
            if r + g + b >= 600:
                cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2,
                            cv2.LINE_AA)

            clicked = False

        # Break the loop when user hits 'esc' key
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
