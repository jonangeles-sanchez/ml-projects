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

from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

from stockpred.datamodel import (
    read_data, plot_price_history, sort_data,
    filter_data, normalize, predict_from_sample, plot_results, prepare_plot
)


def create_model(x_train, y_train):
    model = Sequential()
    model.add(
        LSTM(
            units=50,
            return_sequences=True,
            input_shape=(x_train.shape[1], 1)
        )
    )
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=5, batch_size=1)
    return model


def prepare_data():
    # Read the dataset
    raw_data = read_data()
    # Analyze the closing prices from dataframe
    historical_data = plot_price_history(raw_data)
    # Sort the dataset on datetime
    sorted_data, empty_data = sort_data(historical_data)
    # Filter Date and Close columns
    data = filter_data(sorted_data, empty_data)
    # Normalize the new filtered dataset
    x_train, y_train, valid = normalize(data)
    return data, x_train, y_train, valid


if __name__ == '__main__':
    """Main entry point of stockpred"""
    data, x_train_data, y_train_data, valid_data = prepare_data()
    # Build lstm Model
    lstm_model = create_model(x_train_data, y_train_data)
    # Take a sample of a dataset to make stock price predictions using the model
    closing_price = predict_from_sample(lstm_model, data, valid_data)
    # Save the LSTM model
    lstm_model.save("saved_fb_lstm_model.h5")
    # Visualize the predicted stock costs with actual stock costs
    train_data, valid_data_df = prepare_plot(closing_price, data)
    plot_results(train_data, valid_data_df)
