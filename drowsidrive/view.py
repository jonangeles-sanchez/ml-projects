"""The view of The Driver Drowsiness Detection System
-----------------------------

About this Module
------------------
The goal of this module is to be used as the front end of the Driver Drowsiness
Detection System.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-06-12"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import cv2

from drowsidrive.config import DrowsiDriveConfig
from drowsidrive.model import DrowsiDrive


class DrowsiDriveView:
    """The frontend application of the Drowsiness Detection System"""

    def __init__(self, ddrive: DrowsiDrive):
        """Initialize the model and opencv video capture

        :param ddrive: a DrowsiDrive object
        """
        self.ddrive = ddrive
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.config = DrowsiDriveConfig()

        # Prepare video capture
        fcc = cv2.CascadeClassifier(str(self.config.FACE_CCLASS_PATH))
        lcc = cv2.CascadeClassifier(str(self.config.LEFT_EYE_CCLASS_PATH))
        rcc = cv2.CascadeClassifier(str(self.config.RIGHT_EYE_CCLASS_PATH))
        self.ddrive.set_classifiers(fcc, lcc, rcc)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    def main_loop(self):
        """Detect drowsy faces for all captured images during the run"""
        while True:
            image, gray, height, width = self.capture_image()
            drowsy_state = self.get_drowsy_state(image, gray)
            self.ddrive.handle_score(drowsy_state)
            self.output_state(drowsy_state, image, height, width)
            cv2.imshow('image', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def capture_image(self):
        """Capture an image and extract relevant information from it

        :return: the image captured, its gray scale colored version, its height
                  and its width
        """
        image = self.cap.read()[1]
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image, gray, height, width

    def get_drowsy_state(self, image, gray):
        """Get the drowsy state of the person

        :param image: the current image
        :param gray: the gray-scale current image
        :return: the current drowsiness state of a person on the image
        """
        faces, left_eyes, right_eyes = self.ddrive.detect_faces_parts(gray)
        face_index = self.get_first_face(faces, left_eyes, right_eyes, image)

        if face_index == -1:
            return self.config.NO_FACE

        r_state = l_state = 1
        if len(right_eyes) > face_index:
            r_state = self.get_eye_state(image, right_eyes[face_index])
        if len(left_eyes) > face_index:
            l_state = self.get_eye_state(image, left_eyes[face_index])

        if r_state + l_state == 0:
            return self.config.BOTH_EYES_CLOSED
        else:
            return self.config.EYE_OPEN

    def get_first_face(self, faces, left_eyes, right_eyes, image):
        """Get the first face detected in current image

        :param faces: all faces detected in current image
        :param left_eyes: all left eyes detected in current image
        :param right_eyes: all right eyes detected in current image
        :param image: the current image
        :return: the index of the first face, -1 if no face was detected
        """
        for i, flr in enumerate(zip(faces, left_eyes, right_eyes)):
            f, l, r = flr
            if len(f) == 4 and len(l) == 4 and len(r) == 4:
                self.draw_contour(f, image)
                self.draw_contour(l, image)
                self.draw_contour(r, image)
                return i
        return -1

    def draw_contour(self, zone_coord, image):
        """Draw a rectangle around a zone in an image

        :param zone_coord: the coord of the zone to draw
        :param image: the image where the zone is located
        """
        x, y, w, h = zone_coord
        cv2.rectangle(image, (x, y), (x + w, y + h), self.white, 1)

    def get_eye_state(self, image, eye_coord):
        """Process the eye submitted

        :param image: the current image
        :param eye_coord: the eye coordinates in the image
        :return: 0 on eye close, 1 on eye open
        """
        eye_sliced = self.ddrive.get_slice(image, eye_coord)
        eye_grayed = cv2.cvtColor(eye_sliced, cv2.COLOR_BGR2GRAY)
        eye_resized = cv2.resize(eye_grayed, (24, 24))
        eye_state = self.ddrive.get_state(eye_resized)
        return eye_state

    def output_state(self, drowsy_state, image, height, width):
        """Output the state of the current image

        :param drowsy_state: the drowsiness state of the person
        :param image: the current image
        :param height: the height of the image
        :param width: the width of the image
        """
        if self.ddrive.score > 10:
            self.trigger_alert(image, height, width)
        self.draw_state_label(drowsy_state, image, height)

    def trigger_alert(self, image, height, width):
        """The person is feeling sleepy so we beep the alarm

        :param image: the current image
        :param height: the height of the image
        :param width: the width of the image
        """
        cv2.imwrite('image.jpg', image)
        self.ddrive.play_alert_sound()
        self.draw_alert_frame(image, height, width)

    def draw_alert_frame(self, image, height, width):
        """Draw a thick red frame around the current image

        :param image: the current image
        :param height: the height of the image
        :param width: the width of the image
        """
        self.ddrive.thickness = 2
        cv2.rectangle(
            image, (0, 0), (width, height), self.white, self.ddrive.thickness
        )

    def draw_state_label(self, drowsy_state, image, height):
        """Draw the current image state

        :param drowsy_state: the drowsiness state of the person
        :param image: the current image
        :param height: the height of the image
        """
        cv2.rectangle(
            image, (0, height - 50), (400, height), self.black,
            thickness=cv2.FILLED
        )
        cv2.putText(
            image, self.config.LABELS[drowsy_state], (10, height - 20),
            self.font, 1, self.white, 1, cv2.LINE_AA
        )
        cv2.putText(
            image, 'Score: ' + str(self.ddrive.score), (250, height - 20),
            self.font, 1, self.white, 1, cv2.LINE_AA
        )
