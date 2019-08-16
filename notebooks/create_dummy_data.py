from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer
import pmdarima as pm

import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")


def parse_month(x):
    year = int(x.split("-")[0])
    month = int(x.split("-")[1])
    day = 1
    return datetime.date(day=day, month=month, year=year)


df = pd.read_csv('notebooks/shampoo-sales.csv')

df.Sales = df.Sales.astype(float)
df.Date = df.Date.apply(parse_month)
df = df.set_index("Date")
df = df.dropna()
