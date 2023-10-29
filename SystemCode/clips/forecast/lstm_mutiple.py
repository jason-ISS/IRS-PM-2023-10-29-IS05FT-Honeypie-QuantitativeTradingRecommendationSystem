import joblib
import numpy as np
import pandas as pd
import time
from forecast.utils import *
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


# prepare data
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    agg_list = []

    for i in range(len(df) - n_in - n_out + 1):
        temp = df.iloc[i:i + n_in + n_out, :]
        agg_list.append(temp)

    agg = np.stack(agg_list)

    if dropnan:
        mask = ~np.isnan(agg).any(axis=(1, 2))
        agg = agg[mask]

    X = agg[:, :n_in, :]
    y = agg[:, n_in:, :]

    return X, y


def dataProcessing(data, step_in, step_out=1):
    data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
    data.set_index('trade_date', inplace=True)
    data.sort_index(inplace=True)
    train_data = data[['high', 'low', 'close', 'change']]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = scaler.fit_transform(train_data)
    X, y = series_to_supervised(train_data, step_in, step_out)
    return X, y, scaler


# save model
def train(train_data, step_in, model_path='../model/ts_code'):
    step_out = 1
    X, y, scaler = dataProcessing(train_data, step_in, step_out)
    y = y.reshape((-1, y.shape[-1]))
    # transfer data format [samples, timesteps, features]
    n_features = y.shape[1]
    # construct LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(n_features))
    model.compile(loss='mae', optimizer='adam')

    # train model
    history = model.fit(X, y, epochs=15, batch_size=128, verbose=2, validation_data=(X, y), shuffle=True)
    # showTrain(history)
    # save model
    model.save(f'{model_path}.h5')
    joblib.dump(scaler, f'{model_path}.pkl')
    return model


# forecast with trained data
def predict(input_data, step_in, n_step, model_path='../model/lstm.h5'):
    # load model
    model = load_model(f'{model_path}.h5')
    scaler = joblib.load(f'{model_path}.pkl')
    # prepare input data
    input_data = input_data[['high', 'low', 'close', 'change']].iloc[-step_in:, :].values
    input_data = scaler.transform(input_data)
    input_data = input_data.reshape((1, step_in, -1))
    # initialize forecast list
    predictions = []
    for _ in range(n_step):
        prediction = model.predict(input_data)
        predictions.append(prediction[0])
        prediction = prediction.reshape(1, 1, -1)
        input_data = np.concatenate((input_data[:, 1:, ], prediction), axis=1)
    # return result
    predictions = np.array(predictions)
    predictions = scaler.inverse_transform(predictions)
    print('预测完毕')
    return predictions


def fun(ts_code):
    n_step = 7
    count = 250
    step_in = 3
    data = pd.read_csv(f'../data/{ts_code}.csv').iloc[:count, :].iloc[::-1]
    # convert data in the order of time
    data = data.iloc[::-1]
    if data.shape[0] < n_step:
        return
    train_data = data.iloc[:-n_step, :]
    test_data = data.iloc[-n_step:, :]

    model = train(train_data, step_in, f"model/{ts_code}")
    pred = predict(train_data, step_in=step_in, n_step=n_step, model_path=f"model/{ts_code}")
    print(count)
    # evaluate(test_data['close'], pred)
    test_data = test_data[['high', 'low', 'close', 'change']].iloc[:n_step, ]
    showTruePred(test_data, pred)


if __name__ == "__main__":
    start_time = time.time()
    df = pd.read_csv('../data/allStock.csv')
    for ts_code in df['ts_code']:
        fun(ts_code)
    #
    # fun('000001.SZ')

    print(time.time() - start_time)
