import pandas as pd
from forecast.utils import *
from prophet import Prophet


def forecast(data, steps_to_predict):
    data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
    data = data[['trade_date', 'close']]
    data.columns = ['ds', 'y']

    # initialize model
    model = Prophet(daily_seasonality=True)
    model.fit(data)

    # construct future_dataframe
    future = model.make_future_dataframe(periods=steps_to_predict)
    forecast = model.predict(future)

    # get the prediction
    predictions = forecast[['ds', 'yhat']].tail(steps_to_predict)

    return predictions

if __name__ == "__main__":
    n_step = 7
    count = 200
    data = pd.read_csv('../data/000001.SZ.csv').iloc[:count, :]
    # convert data
    data = data.iloc[::-1]

    train_data = data.iloc[:-n_step, :]
    test_data = data.iloc[-n_step:, :]
    pred = forecast(train_data, n_step)
    print(count)
    evaluate(test_data['close'], pred['yhat'])
    showTruePred(test_data['close'], pred['yhat'])
