import argparse
import pandas as pd
from stax import TimeSeries
from datetime import datetime
import matplotlib.pyplot as plt
import json
import numpy as np

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf, pacf
from matplotlib.pyplot import cm

# Read data
df = pd.read_csv('data/US-beer-sales.csv')
# Make datetime index
df = df.set_index("Date")
df.index = pd.to_datetime(df.index)
# Select series of interest
series = df.Sales
# Build time series object
ts = TimeSeries(series, "monthly", train_test_split=0.8)
ts.train_models()

color = iter(cm.rainbow(np.linspace(0, 1, 7)))
results = ts.experiment_results
# series.plot()
split = results["meta"]["train_test_split_index"]
plt.title("US Beer Sales Model Evaluation")
c = next(color)
plt.plot(series.index[:split], series.values[:split], c=c, alpha=0.4)
for model in results["models"]:

    p = results["models"][model]["test_predictions"]
    # print(p)
    plt.plot(series.index[split:], p, label=model, c=c)

    # Confidence Intervals
    if results["models"][model]["test_confidence_intervals"] != None:
        lower = [
            i["lower"]
            for i in results["models"][model]["test_confidence_intervals"]
        ]
        upper = [
            i["upper"]
            for i in results["models"][model]["test_confidence_intervals"]
        ]
        plt.fill_between(series.index[split:],
                         lower,
                         upper,
                         alpha=0.5,
                         color=c,
                         label=model + "confidence_intervals")
        c = next(color)

plt.plot(series.index[split:], series.values[split:], label="original")
plt.axvline(x=series.index[split], color="black", alpha=0.3, linestyle='--')
plt.legend()
plt.show()