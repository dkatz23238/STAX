import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer
import pmdarima as pm

import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")

df = pd.read_csv('airline-passengers.csv')
df.Month = pd.to_datetime(df.Month)
df = df.set_index(df.Month)[["Passengers"]]

start = round(df.shape[0] * 0.9)

train = df.Passengers.iloc[0:start]
test = df.Passengers.iloc[start:]

horizon = (len(test))

model = ExponentialSmoothing(
    train,
    seasonal='mul',
    seasonal_periods=12,
).fit()
pred = model.predict(start=test.index[0], end=test.index[-1])

plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='best')
plt.show()