import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
from tbats import TBATS

# All functions called train_* return a tuple of the model,
# the predictions on test set, the confidence intervals if
# they are present, and finally a list of dicts containing
# evaluation metrics

# best_model, pred, conf, metrics


def train_arima(ts):
    '''Returns the model, pred, conf, and metrics '''

    if ts.seasonal_N == None:
        stepwise_fit = pm.auto_arima(
            ts.train,
            start_p=1,
            start_q=1,
            max_p=3,
            max_q=3,
            m=2,
            start_P=0,
            seasonal=True,
            d=1,
            D=1,
            trace=False,
            error_action=
            'ignore',  # don't want to know if an order does not work
            suppress_warnings=True,  # don't want convergence warnings
            stepwise=True,
            verbose=False)  # set to stepwise
    else:
        stepwise_fit = pm.auto_arima(
            ts.train,
            start_p=1,
            start_q=1,
            max_p=3,
            max_q=3,
            m=ts.seasonal_N,
            start_P=0,
            seasonal=True,
            d=1,
            D=1,
            trace=False,
            error_action=
            'ignore',  # don't want to know if an order does not work
            suppress_warnings=True,  # don't want convergence warnings
            stepwise=True,
            verbose=False)  # set to stepwise

    horizon = ts.test.shape[0]
    MAPE = np.round(
        mean_absolute_error(stepwise_fit.predict(horizon), ts.test) /
        ts.test.mean(), 4)

    pred, conf = stepwise_fit.predict(horizon, return_conf_int=True)
    metrics = [{"mean_absolute_percent_error": MAPE}]
    return stepwise_fit, pred, conf, metrics


def train_expsmoothing(ts):
    parameter_space = {
        "trend": ['add', 'mul'],
        "seasonal": ['add', 'mul'],
        "seasonal_periods": [12],
    }

    preds = []

    for t in parameter_space["trend"]:
        for s in parameter_space["seasonal"]:
            try:
                model = ExponentialSmoothing(
                    ts.train,
                    trend=t,
                    seasonal=s,
                    seasonal_periods=ts.seasonal_N,
                ).fit()
                pred = model.predict(start=ts.test.index[0],
                                     end=ts.test.index[-1])
                preds.append({
                    "name":
                    t + s,
                    "mae":
                    mean_absolute_error(ts.test.values, pred.values),
                    "model":
                    model
                })
            except:
                pass
    best_model = sorted(preds, key=lambda x: x["mae"])[0]["model"]

    pred = best_model.predict(start=ts.train_test_split,
                              end=ts.train_test_split + ts.test.shape[0] - 1)
    mabe = mean_absolute_error(ts.test, pred)
    MAPE = np.round(mabe / ts.test.mean(), 4)

    metrics = [{"mean_absolute_percent_error": MAPE}]
    return best_model, pred, None, metrics


def train_tbats(ts):
    parameter_space = {
        "seasonal_period": [[6, 15], [12, 15], [6, 30], [12, 30]],
        "use_box_cox": [True, False],
        "use_arma_errors": [True, False],
    }

    results = []

    for sp in parameter_space["seasonal_period"]:
        for bx in parameter_space["use_box_cox"]:
            for ae in parameter_space["use_arma_errors"]:

                estimator = TBATS(use_box_cox=bx,
                                  use_arma_errors=ae,
                                  seasonal_periods=sp)
                horizon = len(ts.test.values)
                model = estimator.fit(ts.train.values)
                pred, conf = model.forecast(steps=horizon,
                                            confidence_level=0.95)
                mape = mean_absolute_error(ts.test.values,
                                           pred) / ts.test.values.mean()
                conf = list(zip(conf["lower_bound"], conf["upper_bound"]))
                results.append({
                    "mape": mape,
                    "model": model,
                    "pred": pred,
                    "conf": conf,
                    "parameters": {
                        "seasonal_period": sp,
                        "use_box_cox": bx,
                        "use_arma_errors": ae
                    }
                })

    best_results = sorted(results, key=lambda x: x["mape"])[0]
    model = best_results["model"]
    pred = best_results["pred"]
    conf = best_results["conf"]
    metrics = [{"mean_absolute_percent_error": best_results["mape"]}]

    return model, pred, conf, metrics
