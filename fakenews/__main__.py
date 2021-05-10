"""The main of the Fake News Detector App
-----------------------------

Project structure
-----------------
*fakenews/*
    **__main__.py**:
        The main of the Fake News Detector App

About this Module
------------------
This module is the main entry point of the Fake News Detector App.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-05-10"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def read():
    # Read the data
    data = pd.read_csv('./news.csv')
    return data


def get_data():
    # Get the labels
    labels = df.label

    # Split the dataset
    x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(
        df['text'], labels,
        test_size=0.2,
        random_state=7)
    return x_train_data, x_test_data, y_train_data, y_test_data


def vectorize():
    # Initialize a TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Fit and transform train set, transform test set
    tfidf_train_data = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test_data = tfidf_vectorizer.transform(x_test)
    return tfidf_train_data, tfidf_test_data


def classify():
    # Initialize a PassiveAggressiveClassifier
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    # Predict on the test set and calculate accuracy
    y_pred_data = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score * 100, 2)}%')
    return y_pred_data


if __name__ == '__main__':
    """Main entry point of fakenews"""
    df = read()
    x_train, x_test, y_train, y_test = get_data()
    tfidf_train, tfidf_test = vectorize()
    y_pred = classify()

    # Build confusion matrix
    print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))
