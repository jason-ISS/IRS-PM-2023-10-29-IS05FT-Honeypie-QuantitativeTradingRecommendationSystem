import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def getData(filePath:str):
    if filePath.endswith('.csv'):
        data = pd.read_csv(filePath)
    elif filePath.endswith('.xlsx') or filePath.endswith('.xls'):
        data = pd.read_excel(filePath)
    else:
        raise Exception('need .csv or excel file')
    data.drop(data.columns[0], axis=1, inplace=True)
    data = data.iloc[:10000, ]
    print(data.shape)
    return data.iloc[:500,:]

def getData1(filePath:str):
    if filePath.endswith('.csv'):
        data = pd.read_csv(filePath)
    elif filePath.endswith('.xlsx') or filePath.endswith('.xls'):
        data = pd.read_excel(filePath)
    else:
        raise Exception('need .csv or excel file')
    data.drop(data.columns[0], axis=1, inplace=True)
    cols = data.columns.tolist()
    cols.append(cols.pop(0))
    data = data.reindex(columns=cols)
    data = data.iloc[:10000, ]
    print(data.shape)
    return data.iloc[:500,:]

def prepare_data(X, y, stepIn, stepOut):
    newX, newy = [], []
    for i in range(X.shape[0] - stepIn - stepOut):
        newX.append(X[i:i + stepIn, ])
        if y.any() is not None:
            newy.append(y[i + stepIn:i + stepIn + stepOut, ])
    return np.array(newX), np.array(newy)

def prepare_y(y, stepIn, stepOut):
    newX, newy = [], []
    for i in range(y.shape[0] - stepIn - stepOut):
        newX.append(y[i:i + stepIn])
        newy.append(y[i + stepIn:i + stepIn + stepOut,-1])
    return np.array(newX), np.array(newy)

def splitTrainTest(values, ration):
    values = np.array(values)
    n_train_time = int(values.shape[0] * ration)
    train = values[:n_train_time, :]
    test = values[n_train_time:, :]
    return train, test


def spliteAndNormalizeXy(trainXy, testXy):
    trainX, trainy = trainXy[:, :-1], trainXy[:, -1]
    testX, testy = testXy[:, :-1], testXy[:, -1]
    scalerX = MinMaxScaler()
    scalery = MinMaxScaler()
    trainX = scalerX.fit_transform(trainX)
    testX = scalerX.transform(testX)
    trainy = scalery.fit_transform(trainy.reshape(-1, 1))
    testy = scalery.transform(testy.reshape(-1, 1))
    # joblib.dump(scalery, f'../model/scaler/lstm_scalery.pkl')
    return trainX, trainy, testX, testy, scalerX, scalery


def spliteAndNormalizeY(trainy, testy):
    scaler = MinMaxScaler()
    trainy = scaler.fit_transform(trainy)
    testy = scaler.transform(testy)
    # joblib.dump(scaler, f'../model/scaler/lstm_scaler.pkl')
    return trainy, testy, scaler
