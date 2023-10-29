import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tensorflow as tf

def init():
    """Initialization, solving various issues with plotting and logging."""
    import matplotlib as mpl
    import warnings
    # Symbol display issue
    plt.rcParams['axes.unicode_minus'] = False
    # Displaying Chinese characters issue
    mpl.rcParams['font.family'] = 'SimHei'
    # Ignoring non-critical logs
    warnings.filterwarnings('ignore')
    # Fixing the random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def mae(y_true, y_pred):
    """
    Calculate MAE (Mean Absolute Error).
    """
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    """
    Calculate MAPE (Mean Absolute Percentage Error).
    """
    return np.mean(np.abs((y_true - y_pred) / y_true+0.00000001)) * 100

def mse(y_true, y_pred):
    """
    Calculate MSE (Mean Squared Error).
    """
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    """
    Calculate RMSE (Root Mean Squared Error).
    """
    return np.sqrt(mse(y_true, y_pred))

def r2(y_true, y_pred):
    """
    Calculate R2 (Coefficient of Determination).
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 0.00000001
    return 1 - (ss_res / ss_tot)

def rmpe(y_true, y_pred):
    """
    Calculate RMPE (Root Mean Percentage Error).
    """
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2)) * 100

def evaluate(yTrue, yPredict):
    """Calculate various evaluation metrics."""
    yTrue = np.array(yTrue)
    yPredict = np.array(yPredict)
    yPredict = yPredict.reshape(-1)
    from sklearn.metrics import mean_squared_error, r2_score
    from scipy.stats import pearsonr
    MSE = mean_squared_error(yTrue, yPredict)
    RMSE = round(np.sqrt(MSE), 2)
    MAPE = round(mape(yTrue, yPredict), 2)
    pearsonrValue = round(pearsonr(yTrue, yPredict)[0], 2)
    r2 = round(r2_score(yTrue, yPredict), 2)

    print(f'Test RMSE: {RMSE}')
    print(f'Test MAPE: {MAPE}')
    print(f'Pearson coefficient: {pearsonrValue}')
    print(f'Coefficient of determination (R2): {r2}')
    return {'RMSE': rmse, 'MAPE': mape, 'Person': pearsonrValue, 'R2': r2}

def evaluate2(yTrue, yPredict):
    """Calculate various evaluation metrics."""
    yTrue = np.array(yTrue)
    yPredict = np.array(yPredict)
    MAE = round(mae(yTrue, yPredict), 3)
    MAPE = round(mape(yTrue, yPredict), 2)
    MSE = round(mse(yTrue, yPredict), 2)
    RMPE = round(rmpe(yTrue, yPredict))
    RMSE = round(rmse(yTrue, yPredict), 3)
    R2 = round(r2(yTrue, yPredict) * 100, 2)
    result = {'MAE': MAE, 'MAPE': MAPE, 'MSE': MSE, 'RMPE': RMPE, 'RMSE': RMSE, 'R2': R2}
    print(result)
    return result

def showTrain(history):
    """Display training process."""
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

def showTruePred(yTrue:pd.DataFrame, yPredict):
    """Display true and predicted values."""
    if yTrue.shape[1]>1:
        nameList = yTrue.columns
        for i, colName in enumerate(nameList):
            plt.plot(yTrue.loc[:,colName].values, label=f'yTrue-{nameList[i]}')
            plt.plot(yPredict[:,i], label=f'yPre-{nameList[i]}')
            plt.legend()
            plt.show()
    else:
        yTrue = np
