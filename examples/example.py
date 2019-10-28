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

# Read data
df = pd.read_csv('US-beer-sales.csv')
# Make datetime index
df = df.set_index("Date")
df.index = pd.to_datetime(df.index)
# Select series of interest
series = df.Sales
# Build time series object
ts = TimeSeries(series, "monthly", train_test_split=0.6)

plt.style.use("ggplot")
f, conf = acf(ts.series, alpha=0.05)
x = range((len(f)))
f1 = [i[0] for i in conf]
f2 = [i[1] for i in conf]
plt.fill_between(x, f1, f2, alpha=0.8)
plt.plot(x, f)