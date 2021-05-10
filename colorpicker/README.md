# The Color Detection using Pandas & OpenCV

![img.png](img.png)

Human eyes and brains work together to translate light into color. Light 
receptors that are present in our eyes transmit the signal to the brain. Our 
brain then recognizes the color. Since childhood, we have mapped certain 
lights with their color names. We will be using the somewhat same strategy 
to detect color names.

## About the project

In this color detection Python project, we are going to build an application 
through which you can automatically get the name of the color by clicking on 
them. So for this, we will have a data file that contains the color name and 
its values. Then we will calculate the distance from each color and find the 
shortest one.

Colors are made up of 3 primary colors; red, green, and blue. In computers, 
we define each color value within a range of 0 to 255. So in how many ways 
we can define a color? The answer is 256*256*256 = 16,581,375. There are 
approximately 16.5 million different ways to represent a color. In our 
dataset, we need to map each color’s values with their corresponding names. 
But, we don’t need to map all the values here.

## Prerequisite

Before starting with this Python project we needed to familiarize with the 
computer vision library of Python that is OpenCV.

OpenCV, Pandas, and numpy are the Python packages that were necessary for 
this project in Python. To install them, we simply ran this command in the 
terminal:
    
    pip install opencv-python numpy pandas

## Project Plan

- [x] Code the whole project
- [x] Document the code
