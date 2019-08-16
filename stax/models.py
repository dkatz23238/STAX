import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import median_absolute_error


def train_arima(ts):
    '''Returns the model, pred, conf, and MAPE '''
    stepwise_fit = pm.auto_arima(
        ts.train,
        start_p=1,
        start_q=1,
        max_p=3,
        max_q=3,
        m=4,
        start_P=0,
        seasonal=True,
        d=1,
        D=1,
        trace=False,
        error_action='ignore',  # don't want to know if an order does not work
        suppress_warnings=True,  # don't want convergence warnings
        stepwise=True,
        verbose=False)  # set to stepwise

    horizon = ts.test.shape[0]
    MAPE = np.round(
        median_absolute_error(stepwise_fit.predict(horizon), ts.test) /
        ts.test.mean(), 4)

    pred, conf = stepwise_fit.predict(horizon, return_conf_int=True)

    return stepwise_fit, pred, conf, MAPE


def train_expsmoothing(ts):
    model = ExponentialSmoothing(
        ts.train,
        seasonal='mul',
        seasonal_periods=12,
    ).fit()

    pred = model.predict(start=ts.train_test_split,
                         end=ts.train_test_split + ts.test.shape[0] - 1)
    mabe = median_absolute_error(ts.test, pred)
    MAPE = np.round(mabe / ts.test.mean(), 4)

    return model, pred, None, MAPE
