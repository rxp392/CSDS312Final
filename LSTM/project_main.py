# Reference: https://pub.towardsai.net/time-series-prediction-of-bitcoin-price-using-lstms-b8a6455d8143

import pandas as pd
import numpy as np
import datetime
import time
import math

import matplotlib.pyplot as plt
import plotly.express as px

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def change_timestamp(ts):
    digit_count = len(str(ts))
    if digit_count == 12:
        return (datetime.datetime.utcfromtimestamp(ts)).strftime('%Y-%m-%d %H:%M:%S')
    else:
        return (datetime.datetime.utcfromtimestamp(ts / 1000)).strftime('%Y-%m-%d %H:%M:%S')


def plot_data(df):
    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), (df['low'] + df['high']) / 2.0)
    plt.xticks(range(0, df.shape[0], 1000), df['dt_correct'].loc[::1000], rotation=45)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price ($)', fontsize=18)
    plt.title("BTC Price History")
    plt.savefig("BTC_price_data.png")
    plt.clf()


def create_lagging_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


if __name__ == "__main__":
    np.random.seed(12345)

    df = pd.read_csv("BTCUSDT_Binance_futures_data_hour.csv", skiprows=1, low_memory=False)
    df_copy = df.copy()

    # preprocess date
    df_copy['unix_count'] = df.unix.apply(lambda x: len(str(x)))
    df_copy['dt_correct'] = df.unix.apply(lambda x: change_timestamp(x))
    df_copy['dt'] = pd.to_datetime(df_copy.dt_correct.values)
    df_copy['hour'] = df_copy.dt.apply(lambda x: x.hour)
    df_copy['week_day'] = df_copy.dt.apply(lambda x: x.weekday())
    df_copy.sort_values(by=['unix'], ascending=[True], inplace=True)
    print(df_copy.dtypes)
    # plot_data(df_copy)

    df_work = df_copy[['dt', 'hour', 'week_day', 'close', 'Volume BTC']]

    # normalize the data
    X = df_work[['hour', 'week_day', 'Volume BTC', 'close']]
    Y = df_work[['close']]

    f_transformer = preprocessing.MinMaxScaler((-1, 1))
    f_transformer = f_transformer.fit(X)

    cnt_transformer = preprocessing.MinMaxScaler((-1, 1))
    cnt_transformer = cnt_transformer.fit(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

    X_train_trans = f_transformer.transform(X_train)
    X_test_trans = f_transformer.transform(X_test)

    y_train_trans = cnt_transformer.transform(y_train)
    y_test_trans = cnt_transformer.transform(y_test)
    
    # Lagging Dataset
    time_steps = 24

    X_train_f, y_train_f = create_lagging_dataset(X_train_trans, y_train_trans, time_steps)
    X_test_f, y_test_f = create_lagging_dataset(X_test_trans, y_test_trans, time_steps)

    # the model
    model = keras.Sequential()
    model.add(keras.Input(shape=(X_train_f.shape[1], X_train_f.shape[2])))
    model.add(keras.layers.LSTM(300, return_sequences=False, activation='tanh'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    model.save("btc_model")

    # Results
    hist = model.fit(X_train_f, y_train_f, batch_size=200, epochs=100, shuffle=False, validation_split=0.1)
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("BTC_Training_and_Validation_loss.png")
    plt.clf()

    y_pred = model.predict(X_test_f)

    y_test_inv = cnt_transformer.inverse_transform(y_test_f)
    y_pred_inv = cnt_transformer.inverse_transform(y_pred)
    combined_array = np.concatenate((y_test_inv, y_pred_inv), axis=1)
    combined_array2 = np.concatenate((X_test.iloc[time_steps:], combined_array), axis=1)

    df_final = pd.DataFrame(data=combined_array, columns=["actual", "predicted"])
    print(df_final.head(4))

    from sklearn.metrics import mean_squared_error

    results = model.evaluate(X_test_f, y_test_f)

    print(f"mse: {mean_squared_error(y_test_inv, y_pred_inv)}")
    print(results)

    ##PREPARING DATA FOR PLOTLY

    a = np.repeat(1, len(y_test_inv))
    b = np.repeat(2, len(y_pred_inv))

    df1 = pd.DataFrame(data = np.concatenate((y_test_inv,(np.reshape(a, (-1, 1)))),axis=1), columns=["price","type"])
    df2 = pd.DataFrame(data = np.concatenate((y_pred_inv,(np.reshape(b, (-1, 1)))),axis=1), columns=["price","type"])

    frames = [df1, df2]
    result = pd.concat(frames, ignore_index=False)

    result["type"].replace({1: "actual", 2: "predict"}, inplace=True)
    (result[result.type == "actual"]).head(10)

    fig = px.line(result, x=result.index.values, y="price", color="type", title="Bitcoin Price Prediction")
    fig.write_image("btc_results.png")
