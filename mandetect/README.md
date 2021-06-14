# Real-time Human Detection & Counting

![img.png](img.png)

The crowd counting task is actually a popular research problem. We are now
concerned about safety issues, notably with last year public health issues.
Considering the scenario of a crowded scene during the epidemic: a population
density system analyzes the crowds and triggers a warning to divert the crowds
when their population density exceeds the normal range. This could all be
executed by drones and spares law enforcement officers for more groundwork task.

## About the project

In this project, we have implemented a system which is able to detect and count
human from sequential frames from a specific angle of view of live feed from a
random city web cam.

### Histogram of Oriented Gradient Descriptor

HOG is a feature descriptor used in computer vision and image processing for the
purpose of object detection. This is one of the most popular techniques for
object detection, to our fortune, OpenCV has already been implemented in an
efficient way to combine the HOG Descriptor algorithm with Support Vector
Machine or SVM.

## Prerequisite

To understand the project it requires to have basic knowledge of python
programming and the OpenCV library. We will needed the following libraries:

    OpenCV: A strong library used for machine learning
    Imutils: To Image Processing
    Numpy: Used for Scientific Computing. Image is stored in a numpy array.
    Argparse: Used to give input in command line.

To install the required library, run the following code in your terminal.

    pip install opencv-python
    pip install imutils
    pip install numpy

Or you may have installed all this installing the conda env from this
ml-projects environment.yml file which is largely enough.

## Project Plan

- [ ] Import the libraries
- [ ] Create a model which will detect Humans
- [ ] Detect functionality
- [ ] Human detector functionality
- [ ] Detect by camera functionality
- [ ] Detect by video path functionality
- [ ] Detect by image path functionality
- [ ] Parsing all arguments
- [ ] The main function