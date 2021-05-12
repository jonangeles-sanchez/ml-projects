"""The main of the Music Genre classifier application
-----------------------------

About this Module
------------------
This module is the main entry point of The main of the Music Genre classifier
application.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-05-11"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import operator
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc


def calculate_distance(instance1, instance2, k):
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2 - mm1).transpose(), np.linalg.inv(cm2)),
                        mm2 - mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance


def get_neighbors(training_set, instance, k):
    distances = []
    for x in range(len(training_set)):
        dist = (
                calculate_distance(training_set[x], instance, k) +
                calculate_distance(instance, training_set[x], k)
        )
        distances.append((training_set[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def nearest_class(neighbors):
    class_vote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in class_vote:
            class_vote[response] += 1
        else:
            class_vote[response] = 1
    sorter = sorted(class_vote.items(), key=operator.itemgetter(1),
                    reverse=True)
    return sorter[0][0]


def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    if len(test_set) > 0:
        return 1.0 * correct / len(test_set)
    else:
        return 0


def load_dataset(filename, split):
    results = []
    tr_set = []
    te_set = []
    with open(filename, 'rb') as f:
        while True:
            try:
                results.append(pickle.load(f))
            except EOFError:
                f.close()
                break

    if split is not None:
        for x in range(len(results)):
            if random.random() < split:
                tr_set.append(results[x])
            else:
                te_set.append(results[x])
    return results, tr_set, te_set


def train():
    directory = "./genres"
    with open("my.dat", 'wb') as f:
        for i, folder in enumerate(next(os.walk(directory))[1]):
            for file in os.listdir(Path(directory, folder)):
                (rate, sig) = wav.read(
                    Path(directory, folder, file).as_posix())
                mfcc_feat = mfcc(sig, rate, winlen=0.020,
                                 appendEnergy=False)
                covariance = np.cov(np.matrix.transpose(mfcc_feat))
                mean_matrix = mfcc_feat.mean(0)
                feature = (mean_matrix, covariance, i+1)
                pickle.dump(feature, f)

    dataset, training_set, test_set = load_dataset("my.dat", 0.9)
    size = len(test_set)
    predictions = []
    for x in range(size):
        predictions.append(
            nearest_class(get_neighbors(training_set, test_set[x], 5)))
    accuracy1 = get_accuracy(test_set, predictions)
    print(f"Accuracy: {100*accuracy1}")


def valid():
    dataset = load_dataset("my.dat", 0)[0]
    results = defaultdict(str)

    for i, folder in enumerate(next(os.walk("./genres"))[1]):
        results[i + 1] = folder

    rate, sig = wav.read("sample2.wav")
    mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
    covariance = np.cov(np.matrix.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(0)
    feature = (mean_matrix, covariance, 0)
    pred = nearest_class(get_neighbors(dataset, feature, 5))
    print(f"Sample music genre: {results[pred]}")


if __name__ == '__main__':
    """Main entry point of whatmusic"""
    train()
    valid()
