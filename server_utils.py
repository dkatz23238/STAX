import pandas as pd
import json
from stax import TimeSeries
from stax import models

import simplejson
import requests
import os

STAX_BACKEND_API = os.environ.get("STAX_BACKEND_URI")

print("STAX BACKEND URI:")
print(STAX_BACKEND_API)
print("")


def series_to_df(data):
    df = pd.DataFrame(data["data"])
    df.columns = ["Date", data["metadata"]["variable_name"]]
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def run_analysis(request):
    data = json.loads(request.content)
    df = series_to_df(data)

    if not "Date" in df.columns:
        raise Exception("No Date")

    series = df.set_index("Date")[data["metadata"]["variable_name"]]
    series.index = pd.to_datetime(series.index)
    ts = TimeSeries(series, data["metadata"]["frequency"], 0.8)
    ts.calculate_statistics()
    ts.train_models()
    # post results

    return ts


def models_to_JSON(ts, series_id):
    models = []
    for key in ts.experiment_results["models"].keys():
        model = ts.experiment_results["models"][key]
        d = {
            "model_name":
            key,
            "_series":
            series_id,
            "test_predictions":
            model["test_predictions"],
            "test_confidence_intervals":
            model["test_confidence_intervals"],
            "mean_absolute_percent_error":
            model["metrics"][0]["mean_absolute_percent_error"],
            "train_test_split_index":
            ts.train_test_split
        }
        models.append(d)
    return models


def decomposition_to_JSON(ts, series_id):
    decomp = ts.experiment_results["seasonal_decomposition"]
    decomp["_series"] = series_id
    decomp["method"] = "multiplicative"
    return decomp


def autocorrelation_to_JSON(ts, series_id):
    autocorr = ts.experiment_results["autocorrelation"]
    autocorr["_series"] = series_id
    return autocorr


# simplejson.dumps(autocorr, ignore_nan=True)
def run_primary_job(request, experiment_id, user_token):
    print("Running primary Job on TimeSeries")
    HEADERS = {"X-Auth-Token": user_token, "Content-Type": "application/json"}
    # print(request.json())
    series_id = request.json()["_id"]
    ts = run_analysis(request)
    models_json = models_to_JSON(ts, series_id)
    decomp_json = decomposition_to_JSON(ts, series_id)
    autocorr_json = autocorrelation_to_JSON(ts, series_id)
    # return models_json, decomp_json, autocorr_json

    models_ids = []
    for model in models_json:
        model_response = requests.post(f"{STAX_BACKEND_API}/api/models",
                                       headers=HEADERS,
                                       json=model)
        assert model_response.status_code == 200
        models_ids.append(json.loads(model_response.content)["_id"])

    decomp_response = requests.post(f"{STAX_BACKEND_API}/api/decompositions",
                                    headers=HEADERS,
                                    data=simplejson.dumps(decomp_json,
                                                          ignore_nan=True))

    assert decomp_response.status_code == 200
    autocorr_response = requests.post(
        f"{STAX_BACKEND_API}/api/autocorrelations",
        headers=HEADERS,
        data=simplejson.dumps(autocorr_json, ignore_nan=True))
    assert autocorr_response.status_code == 200

    update_experiment_json = {
        "status": "complete",
        "_models": models_ids,
        "_autocorrelation": json.loads(decomp_response.content)["_id"],
        "_decomposition": json.loads(decomp_response.content)["_id"]
    }

    update_experiment_response = requests.put(
        f"{STAX_BACKEND_API}/api/experiments/{experiment_id}",
        json=update_experiment_json,
        headers=HEADERS)

    print(update_experiment_json)

    return True