import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  mean_squared_error

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

def parse_date(date_string):
  return datetime.fromtimestamp(date_string)


df = pd.read_csv("../data/estimated-transaction-volume.csv")
df.Date = pd.to_datetime(df.Date)
df.index = df.Date
df = df[df.index > datetime(2018,1,1)]

scaler = MinMaxScaler(feature_range=(0,10))
df["Vol"] = scaler.fit_transform(df["Vol"].values.reshape(df.shape[0],-1))
del df["Date"]

tr_start = datetime(2016,1,1)
tr_end = datetime(2019,2,1)
te_start = tr_end 
te_end = datetime(2019,3,1)


train = df.loc[tr_start:tr_end] 
test = df.loc[te_start:te_end]

plt.plot(train)
plt.plot(test)
plt.show()


Qs = range(0, 4)
qs = range(0, 4)
Ps = range(0, 4)
ps = range(0, 4)
D=1
d=1

parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)

# Model Selection
results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')
for param in parameters_list:
    try:
        model=sm.tsa.statespace.SARIMAX(
          train,
          order=(param[0], d, param[1]),
          seasonal_order=(param[2], D, param[3], 12),
          mle_regression=True
          ).fit(disp=-1)
    except ValueError:
        print('wrong parameters:', param)
        continue
    
    aic = model.aic

    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param

    results.append([param, model.aic])


result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by = 'aic', ascending=True).head())
print(best_model.summary())

def invboxcox(y,lmbda):
   if lmbda == 0:
      return(np.exp(y))
   else:
      return(np.exp(np.log(lmbda*y+1)/lmbda))


predict = (best_model.predict(test.index[0], test.index[-1]))
pred = scaler.inverse_transform(predict.values.reshape(predict.shape[0],1))
test = scaler.inverse_transform(test.values.reshape(test.shape[0],1))
print('SARIMA model RMSE:{}'.format(mean_squared_error(test,pred)**0.5))
# plt.plot(pred,label="pred")
#plt.plot(test, label="true")
# plt.legend()
# plt.show()
