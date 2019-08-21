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

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def decompose_series(ts):
    d = seasonal_decompose(ts.series, two_sided=False)
    result = {
        "trend": list(d.trend),
        "seasonal": list(d.seasonal),
        "resid": list(d.resid)
    }
    return result


def ACF(ts):
    return list(acf(ts.series))


def PACF(ts):
    return list(pacf(ts.series))