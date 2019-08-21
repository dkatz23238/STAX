import argparse
import pandas as pd
from stax import TimeSeries
from datetime import datetime
import matplotlib.pyplot as plt
import json

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Read data
df = pd.read_csv('sunspot-count.csv')
# Make datetime index
df = df.set_index("Date")
df.index = pd.to_datetime(df.index)
# Select series of interest
series = df.Sunspots
# Build time series object
ts = TimeSeries(series, "monthly", train_test_split=0.95)
# Run the experiments
ts.train_models()

#