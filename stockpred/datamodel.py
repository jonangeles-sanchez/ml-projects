"""The Data Models processing of the Stock Prediction app
-----------------------------

About this Module
------------------
The goal of this module is to prepare data, predict from models and plot data
from prediction results.
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-05-10"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler

rcParams['figure.figsize'] = 20, 10
scaler = MinMaxScaler(feature_range=(0, 1))


def read_data():
    results = pd.read_csv("stock_data.csv")
    results = results[results.Stock == "FB"]
    return results


def plot_price_history(df):
    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df.index = df['Date']
    plt.figure(figsize=(16, 8))
    plt.plot(df["Close"], label='Close Price history')
    return df


def sort_data(df):
    sorted_df = df.sort_index(ascending=True, axis=0)
    empty_df = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
    return sorted_df, empty_df


def filter_data(in_df, out_df):
    for i in range(0, len(in_df)):
        out_df["Date"][i] = in_df['Date'][i]
        out_df["Close"][i] = in_df["Close"][i]
    out_df.index = out_df.Date
    out_df.drop("Date", axis=1, inplace=True)
    return out_df


def normalize(df):
    values = df.values
    train = values[0:int(len(values)*.9), :]
    valid = values[int(len(values)*.9):, :]
    scaled = scaler.fit_transform(values)
    x_train, y_train = [], []
    for i in range(60, len(train)):
        x_train.append(scaled[i - 60:i, 0])
        y_train.append(scaled[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train, valid


def predict_from_sample(model, df, valid):
    in_df = df[len(df) - len(valid) - 60:].values
    in_df = in_df.reshape(-1, 1)
    in_df = scaler.transform(in_df)
    x_test = []
    for i in range(60, in_df.shape[0]):
        x_test.append(in_df[i - 60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    prediction = model.predict(x_test)
    prediction = scaler.inverse_transform(prediction)
    return prediction


def prepare_plot(prediction, df):
    train = df[:int(len(df)*.9)]
    valid = df[int(len(df)*.9):]
    prediction_df = pd.DataFrame(prediction, columns=["Predictions"],
                                 index=valid.index)
    valid_prediction_df = pd.concat([df, prediction_df])
    return train, valid_prediction_df


def plot_results(plot1, plot2):
    plt.plot(plot1["Close"])
    plt.plot(plot2[["Close", "Predictions"]])
    plt.show()
