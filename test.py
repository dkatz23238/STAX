import json
import subprocess
import os

sub_result = subprocess.call(
    "python -m stax data/airline-passengers.csv Passengers monthly 0.8 result.json"
    .split(" "))

result = json.loads(open("./result.json").read())

train_test_split_index = result["meta"]["train_test_split_index"]
arima_mean_absolute_percent_error = result["models"]["ARIMA"]["metrics"][0][
    "mean_absolute_percent_error"]

ets_mean_absolute_percent_error = result["models"]["ExponentialSmoothing"][
    "metrics"][0]["mean_absolute_percent_error"]

assert ets_mean_absolute_percent_error < 0.10
assert ets_mean_absolute_percent_error < 0.10
assert train_test_split_index == 115

os.remove("result.json")