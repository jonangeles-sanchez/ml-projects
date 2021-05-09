"""The main of the Stock Price Prediction app
-----------------------------

About this Module
------------------
This module is the main entry point of The main of the Stock Price Prediction app.

"""

__author__ = "Benoit Lapointe"
__date__ = "2021-05-07"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

rcParams['figure.figsize'] = 20, 10


def read_data():
    results = pd.read_csv("NSE-Tata-Global-Beverages-Limited.csv")
    results.head()
    return results


def plot_price_history(a_df):
    a_df["Date"] = pd.to_datetime(a_df.Date, format="%Y-%m-%d")
    a_df.index = a_df['Date']
    plt.figure(figsize=(16, 8))
    plt.plot(a_df["Close"], label='Close Price history')
    return a_df


def sort_data(df):
    sorted_df = df.sort_index(ascending=True, axis=0)
    empty_df = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
    return sorted_df, empty_df


def filter_data(sorted_df, filtered_df):
    for i in range(0, len(sorted_df)):
        filtered_df["Date"][i] = sorted_df['Date'][i]
        filtered_df["Close"][i] = sorted_df["Close"][i]
    filtered_df.index = filtered_df.Date
    filtered_df.drop("Date", axis=1, inplace=True)
    return filtered_df


def normalize(df, scaler):
    values = df.values
    train = values[0:987, :]
    valid = values[987:, :]
    scaled = scaler.fit_transform(values)
    x_train, y_train = [], []
    for i in range(60, len(train)):
        x_train.append(scaled[i - 60:i, 0])
        y_train.append(scaled[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train, valid, values


def create_model():
    model = Sequential()
    model.add(
        LSTM(
            units=50,
            return_sequences=True,
            input_shape=(x_train_data.shape[1], 1)
        )
    )
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)
    return model


def predict_from_sample(model, new_dataset, valid_data):
    inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
    inputs_data = inputs_data.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)
    x_test = []
    for i in range(60, inputs_data.shape[0]):
        x_test.append(inputs_data[i - 60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    prediction = model.predict(x_test)
    prediction = scaler.inverse_transform(prediction)
    return prediction


def plot_results(prediction):
    train = new_dataset[:987]
    valid = new_dataset[987:]
    prediction_df = pd.DataFrame(prediction, columns=["Predictions"],
                                 index=valid.index)
    merged_data = pd.concat([valid, prediction_df])
    plt.plot(train["Close"])
    plt.plot(merged_data[["Close", "Predictions"]])
    plt.show()


if __name__ == '__main__':
    """Main entry point of stockpred"""
    # Read the dataset
    raw_data = read_data()
    # Analyze the closing prices from dataframe
    historical_data = plot_price_history(raw_data)
    # Sort the dataset on datetime
    sorted_data, empty_data = sort_data(historical_data)
    # Filter Date and Close columns
    new_dataset = filter_data(sorted_data, empty_data)
    # Normalize the new filtered dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_data, y_train_data, valid_data, final_dataset = normalize(
        new_dataset, scaler)
    # Build lstm Model
    lstm_model = create_model()
    # Take a sample of a dataset to make stock price predictions using the model
    closing_price = predict_from_sample(lstm_model, new_dataset, valid_data)
    # Save the LSTM model
    lstm_model.save("saved_lstm_model.h5")
    # Visualize the predicted stock costs with actual stock costs
    plot_results(closing_price)
