import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import median_absolute_error
from stax.models import train_arima, train_expsmoothing, train_tbats
from stax.tools import decompose_series, ACF, PACF


def strftime(dt):
    return dt.isoformat()


def convert_confs(conf):
    if conf is not None:
        return [{"lower": i[0], "upper": i[1]} for i in conf]
    else:
        return None


class TimeSeries(object):
    def __init__(self, series, frequency, train_test_split=0.9):
        # Asserts
        assert frequency in ["daily", "monthly", "quarterly"
                             ], "Frequency must be daily, monthly, or yearly"

        assert type(
            series.index) == pd.DatetimeIndex, "Provide only datetime index"

        assert type(
            series
        ) == pd.Series, "Provide pandas.Series class with datetime index"

        assert series.sum(
        ), "Only numeric data is allowed in time series object"

        self.series = series
        self.frequency = frequency

        if frequency == "daily":
            # Weekly seasonality
            self.seasonal_N = 7
        elif frequency == "monthly":
            # Monthly seasonality
            self.seasonal_N = 12
        elif frequency == "yearly":
            # No seasonality in yearly data
            self.seasonal_N = None

        self.train_test_split = round(series.shape[0] * train_test_split)
        self.train = series.iloc[:self.train_test_split]
        self.test = series.iloc[self.train_test_split:]

        self.experiment_results = {
            "meta": {
                "train_test_split_index": self.train_test_split
            },
            "models": {}
        }

    def calculate_statistics(self):
        sd = "seasonal_decomposition"  #tidy
        self.experiment_results[sd] = decompose_series(self)
        self.experiment_results["autocorrelation"] = {
            "ACF": ACF(self),
            "PACF": PACF(self)
        }

    def train_models(self):
        # Arima models
        m1, p1, conf1, metrics1, oos_pred1, oos_conf1 = train_arima(self)
        conf1 = convert_confs(conf1)

        m2, p2, conf2, metrics2, oos_pred2, oos_conf2 = train_expsmoothing(
            self)
        conf2 = convert_confs(conf2)

        m3, p3, conf3, metrics3, oos_pred2, oos_conf2 = train_tbats(self)
        conf3 = convert_confs(conf3)

        self.experiment_results["models"]["ARIMA"] = {
            "model": m1,
            "test_predictions": p1.tolist(),
            "test_confidence_intervals": conf1,
            "metrics": metrics1
        }

        self.experiment_results["models"]["ExponentialSmoothing"] = {
            "model": m2,
            "test_predictions": p2.tolist(),
            "test_confidence_intervals": conf2,
            "metrics": metrics2
        }

        self.experiment_results["models"]["TBATS"] = {
            "model": m3,
            "test_predictions": list(p3),
            "test_confidence_intervals": list(conf3),
            "metrics": metrics3
        }
