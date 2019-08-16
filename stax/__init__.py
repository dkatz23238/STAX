import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import median_absolute_error
from stax.models import train_arima, train_expsmoothing


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

        self.trained_models = {
            "meta": {
                "train": {
                    "values": self.train.values.tolist()
                },
                "test": {
                    "values": self.test.values.tolist()
                }
            },
            "models": {
                "ARIMA": None,
                "ExponentialSmoothing": None
            }
        }

    def train_models(self):
        m1, p1, conf1, mape1 = train_arima(self)

        m2, p2, conf2, mape2 = train_expsmoothing(self)
        self.trained_models["models"]["ARIMA"] = {
            "model": str(m1),
            "test_predictions": p1.tolist(),
            "test_confidence_intervals": conf1.tolist(),
            "test_mean_absolute_percent_error": mape1
        }

        self.trained_models["models"]["ExponentialSmoothing"] = {
            "model": str(m2),
            "test_predictions": p2.tolist(),
            "test_confidence_intervals": conf2,
            "test_mean_absolute_percent_error": mape2
        }