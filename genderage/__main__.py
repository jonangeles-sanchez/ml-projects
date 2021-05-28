"""The main of Gender and Age Detection with OpenCV
-----------------------------

About this Module
------------------
The goal of this module is to initiate the run of the Gender and Age Detection
and define core components.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-05-28"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import argparse
import os.path
from pathlib import Path

import cv2

# Paths
BASE_PATH = Path(os.path.dirname(__file__))
FACE_PROTO = str(BASE_PATH.joinpath('models', "opencv_face_detector.pbtxt"))
FACE_MODEL = str(BASE_PATH.joinpath('models', "opencv_face_detector_uint8.pb"))
AGE_PROTO = str(BASE_PATH.joinpath('models', "age_deploy.prototxt"))
AGE_MODEL = str(BASE_PATH.joinpath('models', "age_net.caffemodel"))
GENDER_PROTO = str(BASE_PATH.joinpath('models', "gender_deploy.prototxt"))
GENDER_MODEL = str(BASE_PATH.joinpath('models', "gender_net.caffemodel"))

# Model constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = [
    '(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)',
    '(48-53)', '(60-100)'
]
GENDER_LIST = ['Male', 'Female']


def highlight_face(net, a_frame, conf_threshold=0.7):
    """Highlight faces on an image

    It uses a colored frame printing gender and age

    :param net:
    :param a_frame:
    :param conf_threshold:
    :return:
    """

    frame_opencv_dnn = a_frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]
    a_blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300),
                                   [104, 117, 123], True, False)

    net.setInput(a_blob)
    detections = net.forward()
    face_box_list = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_box_list.append([x1, y1, x2, y2])
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0),
                          int(round(frame_height / 150)), 8)
    return frame_opencv_dnn, face_box_list


if __name__ == '__main__':
    """Main entry point of the genderage package"""
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    args = parser.parse_args()

    # Loading the models
    face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

    #
    video = cv2.VideoCapture(args.image if args.image else 0)
    padding = 20
    while cv2.waitKey(1) < 0:
        has_frame, frame = video.read()
        if not has_frame:
            cv2.waitKey()
            break

        result_img, face_boxes = highlight_face(face_net, frame)
        if not face_boxes:
            print("No face detected")

        for face_box in face_boxes:
            face = frame[
                   max(0, face_box[1] - padding): min(face_box[3] + padding,
                                                      frame.shape[0] - 1),
                   max(0, face_box[0] - padding): min(face_box[2] + padding,
                                                      frame.shape[1] - 1)
                   ]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                         MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]
            print(f'Gender: {gender}')

            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGE_LIST[age_preds[0].argmax()]
            print(f'Age: {age[1:-1]} years')

            cv2.putText(result_img, f'{gender}, {age}',
                        (face_box[0], face_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Detecting age and gender", result_img)
