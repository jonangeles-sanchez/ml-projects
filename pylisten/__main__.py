"""The Main of the Speech Emotion Recognition Application
-----------------------------

About this Module
------------------
This module is the main entry point and the model of the Speech Emotion
Recognition Application (SER).
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-05-13"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import glob
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


class PyListen:
    """The model of the Speech Emotion Recognition system"""
    def __init__(self):
        """Initialize emotions in the model"""
        self.emotions = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        self.observed_emotions = ['calm', 'happy', 'fearful', 'disgust']
        # self.observed_emotions = list(self.emotions.values())

    def extract_feature(self, file_name, mfcc, chroma, mel):
        """Extract mfcc, chroma and mel feature from the sound file

        :param file_name: the name of the sound file
        :param mfcc: the Mel-frequency cepstral coefficients
        :param chroma: the chromagram from a waveform or power spectrogram
        :param mel: the mel-scaled spectrogram computed
        :return: the feature extracted
        """
        with soundfile.SoundFile(file_name) as sound_file:
            sound = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            if chroma:
                stft = np.abs(librosa.stft(sound))
            result = np.array([])
            if mfcc:
                mfccs = np.mean(
                    librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=40).T,
                    axis=0)
                result = np.hstack((result, mfccs))
            if chroma:
                chroma = np.mean(
                    librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,
                    axis=0)
                result = np.hstack((result, chroma))
            if mel:
                mel = np.mean(
                    librosa.feature.melspectrogram(sound, sr=sample_rate).T,
                    axis=0)
                result = np.hstack((result, mel))
        return result

    def load_data(self, test_size=0.2):
        """Prepare audio data to be modeled

        :param test_size: the size fraction of the test dataset
        :return:
        """
        x, y = [], []
        data_path = str(Path('data', 'Actor_*', '*.wav'))
        for file in glob.glob(data_path):
            file_name = os.path.basename(file)
            emotion = self.emotions[file_name.split("-")[2]]
            if emotion not in self.observed_emotions:
                continue
            feature = self.extract_feature(file, mfcc=True, chroma=True,
                                           mel=True)
            x.append(feature)
            y.append(emotion)
        return train_test_split(np.array(x), y, test_size=test_size,
                                random_state=9)

    def model(self):
        """Model Speech emotion recognition with MLPClassifier

        :return:
        """
        x_train, x_test, y_train, y_test = self.load_data(test_size=0.1)
        print((x_train.shape[0], x_test.shape[0]))
        print(f'Features extracted: {x_train.shape[1]}')
        model = MLPClassifier(
            alpha=0.01, batch_size=256, epsilon=1e-08,
            hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=800
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        print("Accuracy: {:.2f}%".format(accuracy * 100))


if __name__ == '__main__':
    """The main entry point of the program"""
    PyListen().model()
