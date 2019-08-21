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


df = pd.read_csv('shampoo-sales.csv')

df.Date = df.Date.apply(parse_month)
df = df.set_index("Date")
df = df.dropna()

N = df.shape[0]
start = N - 12

train = df.Sales.values[0:start]
test = df.Sales.values[start:]

horizon = (len(test))

stepwise_fit = pm.auto_arima(
    train,
    start_p=1,
    start_q=1,
    max_p=3,
    max_q=3,
    m=4,
    start_P=0,
    seasonal=True,
    d=1,
    D=1,
    trace=True,
    error_action='ignore',  # don't want to know if an order does not work
    suppress_warnings=True,  # don't want convergence warnings
    stepwise=True,
    verbose=False)  # set to stepwise

from sklearn.metrics import median_absolute_error

MAPE = np.round(
    median_absolute_error(stepwise_fit.predict(horizon), test) / test.mean() *
    100, 4)
print(f"Median Absoloute Percent Error: {MAPE}%")
pred, conf = stepwise_fit.predict(horizon, return_conf_int=True)
plt.figure(figsize=(12, 4))
index_1 = range(len(train))
plt.plot(index_1, train)

index = [i + index_1[-1] + 1 for i in range(len(pred))]
plt.plot(index, pred)
plt.fill_between(index, conf[:, 0], conf[:, 1], color="blue", alpha=0.2)
plt.plot(index, test)
plt.show()