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


def strftime(datetime):
    return datetime.isoformat()


class TimeSeries(object):
    def __init__(self, series, frequency, train_test_split=0.9):
        # Asserts
        assert frequency in ["daily", "monthly", "quarterly"
                             ], "Frequency must be daily, monthly, or yearly"

        assert series.index.dtype == np.dtype(
            'datetime64[ns]'), "Provide date indexed series only"

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

    def calculate_statistcs(self):
        sd = "seasonal_decomposition"  #tidy
        self.experiment_results[sd] = decompose_series(self)
        self.experiment_results["autocorrelation"] = {
            "ACF": ACF(self),
            "PACF": PACF(self)
        }

    def train_models(self):
        # Arima models
        m1, p1, conf1, metrics1 = train_arima(self)

        m2, p2, conf2, metrics2 = train_expsmoothing(self)

        m3, p3, conf3, metrics3 = train_tbats(self)

        self.experiment_results["models"]["ARIMA"] = {
            "model": m1,
            "test_predictions": p1.tolist(),
            "test_confidence_intervals": conf1.tolist(),
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