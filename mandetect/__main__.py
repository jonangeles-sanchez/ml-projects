"""The main of the Real-time Human Detection & Counting
-----------------------------

About this Module
------------------
The goal of this module is to launch the mandetect application from the python
package.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-06-14"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

# Import the libraries
import argparse

import cv2
import imutils
import pafy


def detect(frame):
    """The Detect functionality

    It draws the boxes around people detected and display the count

    :param frame: the current frame to analyze
    """
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(
        frame,
        winStride=(8, 10), padding=(16, 20), scale=1.01
    )

    person = 1
    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, f'person {person}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 255), 1
        )
        person += 1

    cv2.putText(
        frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8,
        (255, 0, 0), 2
    )
    cv2.putText(
        frame, f'Total Persons : {person - 1}', (40, 70),
        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2
    )
    cv2.imshow('output', frame)

    return frame


def human_detector(arg_vars):
    """Human detector functionality

    Start the run by switching between live camera, picture and video url
    capture

    :param arg_vars: argument provided from the parser
    """
    image_path = arg_vars["image"]
    video_path = arg_vars['video']
    camera = arg_vars["camera"]

    writer = None
    if arg_vars['output'] is not None and image_path is None:
        writer = cv2.VideoWriter(
            arg_vars['output'], cv2.VideoWriter_fourcc(*'MJPG'), 10, (600, 600)
        )

    if camera:
        print('[INFO] Opening Web Cam.')
        detect_by_camera(writer)
    elif video_path is not None:
        print('[INFO] Opening Video from path.')
        detect_by_path_video(video_path, writer)
    elif image_path is not None:
        print('[INFO] Opening Image from path.')
        detect_by_path_image(image_path, arg_vars['output'])


def detect_by_camera(writer):
    # Detect by camera functionality
    """Start a capture session from the device camera

    :param writer: writes the image analyzed to the output folder
    """
    video = cv2.VideoCapture(0)
    print('Detecting people...')

    while True:
        check, frame = video.read()

        frame = detect(frame)
        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def detect_by_path_video(src_path, writer):
    """Detect by video path functionality

    :param src_path: the source path of the video file
    :param writer: writes the image analyzed to the output folder
    """
    video = pafy.new(src_path)
    best = video.getbest(preftype="mp4")

    video = cv2.VideoCapture(best.url)
    check, frame = video.read()
    if not check:
        print(
            'Video Not Found. Please Enter a Valid Path (Full path of Video '
            'Should be Provided).'
        )
        return

    frame_count = 0
    print('Detecting people...')
    while video.isOpened():
        # check is True if reading was successful
        check, frame = video.read()

        if check and frame_count == 60:
            frame_count = 0
            frame = imutils.resize(frame, width=min(1920, frame.shape[1]))
            frame = detect(frame)

            if writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1000)
            if key == ord('q'):
                break
        else:
            frame_count += 1
    video.release()
    cv2.destroyAllWindows()


def detect_by_path_image(src_path, output_path):
    """Detect by image path functionality

    :param src_path: the source path of the image provided
    :param output_path: the output_path of the analyzed image
    """
    image = cv2.imread(src_path)

    image = imutils.resize(image, width=min(800, image.shape[1]))

    result_image = detect(image)

    if output_path is not None:
        cv2.imwrite(output_path, result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def args_parser():
    """Parsing all arguments (video, image, camera, output)

    If such arguments are provided
    """
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument(
        "-v", "--video", default=None, help="path to Video File "
    )
    arg_parse.add_argument(
        "-i", "--image", default=None, help="path to Image File "
    )
    arg_parse.add_argument(
        "-c", "--camera", default=False, action='store_true',
        help="Set true if you want to use the camera."
    )
    arg_parse.add_argument(
        "-o", "--output", type=str, help="path to optional output video file"
    )
    arg_vars = vars(arg_parse.parse_args())

    return arg_vars


if __name__ == '__main__':
    """Main entry point of the mandetect package"""
    # Create a model which will detect Humans
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    args = args_parser()
    human_detector(args)
