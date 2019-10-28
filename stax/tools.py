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
    additive = seasonal_decompose(ts.series, model="additive", two_sided=False)
    multiplicative = seasonal_decompose(ts.series,
                                        model="multiplicative",
                                        two_sided=False)
    add_res_mean = np.mean(additive.resid)
    mul_res_mean = np.mean(multiplicative.resid)

    if add_res_mean > mul_res_mean:

        result = {
            "trend": list(multiplicative.trend),
            "seasonal": list(multiplicative.seasonal),
            "resid": list(multiplicative.resid)
        }
    else:
        result = {
            "trend": list(additive.trend),
            "seasonal": list(additive.seasonal),
            "resid": list(additive.resid)
        }

    return result


def ACF(ts):
    return list(acf(ts.series))


def PACF(ts):
    return list(pacf(ts.series))