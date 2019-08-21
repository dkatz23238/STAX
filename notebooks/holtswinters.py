import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer
import pmdarima as pm

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")

df = pd.read_csv('airline-passengers.csv')
df.Date = pd.to_datetime(df.Date)
df = df.set_index(df.Date)[["Passengers"]]

start = round(df.shape[0] * 0.7)
train = df.Passengers.iloc[0:start]
test = df.Passengers.iloc[start - 1:]
horizon = (len(test))

parameter_space = {
    "trend": ['add', 'mul'],
    "seasonal": ['add', 'mul'],
    "seasonal_periods": [12],
}

preds = []

for t in parameter_space["trend"]:
    for s in parameter_space["seasonal"]:
        model = ExponentialSmoothing(
            train,
            trend=t,
            seasonal=s,
            seasonal_periods=12,
        ).fit()
        pred = model.predict(start=test.index[0], end=test.index[-1])
        preds.append({
            "name": t + s,
            "mae": mean_absolute_error(test.values, pred.values)
        })

print(pd.DataFrame(preds).set_index("name"))

#         plt.plot(pred.index, pred, label=s + t, c="purple")
#         plt.legend(loc='best')

# plt.title("Prediction of Airline Passengers by Year")
# plt.plot(train.index, train, label='Train')
# plt.plot(test.index, test, label='Test')
# plt.show()