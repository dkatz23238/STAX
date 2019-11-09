import json
import os

import pandas as pd
import requests
import simplejson

import datetime

from stax import TimeSeries, convert_confs, models

TOKEN = "X8JUN8GP7H"

STAX_BACKEND_API = "https://stax-rest-backend-v2-c2iwk3zleq-uc.a.run.app"
HEADERS = {"X-Auth-Token": TOKEN, "content-type": "application/json"}


def get_experiment(experiment_id):
    r = requests.get(f"{STAX_BACKEND_API}/api/experiments/{experiment_id}",
                     headers=HEADERS)
    return json.loads(r.content)


def put_experiment(experiment_id, data):
    response = requests.put(
        f"{STAX_BACKEND_API}/api/experiments/{experiment_id}",
        headers=HEADERS,
        data=simplejson.dumps(data, ignore_nan=True))
    return response


def series_to_df(data):
    """Returns a pandas.DataFrame"""
    df = pd.DataFrame(data["data"])
    df.columns = ["Date", data["metadata"]["variable_name"]]
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def train_model(series_id, experiment_id, model):
    """Returns a dict"""
    if not (model in ["TBATS", "ARIMA", "ETS"]):
        raise Exception("Only TBATS, ARIMA, or ETS")

    series_res = requests.get(f"{STAX_BACKEND_API}/api/series/{series_id}",
                              headers=HEADERS)

    experiment_res = requests.get(
        f"{STAX_BACKEND_API}/api/experiments/{experiment_id}", headers=HEADERS)

    experiment = json.loads(experiment_res.content)
    series = json.loads(series_res.content)

    var_name = series["metadata"]["variable_name"]

    df = series_to_df(series).set_index("Date")[var_name]
    ts = TimeSeries(df, series["metadata"]["frequency"], 0.8)

    if model == "ARIMA":
        best_model, pred, conf, metrics = models.train_arima(ts)

        d = {
            "model_name":
            "ARIMA",
            "_series":
            series_id,
            "test_predictions":
            list(pred),
            "test_confidence_intervals":
            convert_confs(conf),
            "mean_absolute_percent_error":
            metrics[0]["mean_absolute_percent_error"],
            "train_test_split_index":
            ts.train_test_split
        }

    elif model == "TBATS":
        best_model, pred, conf, metrics = models.train_tbats(ts)

        d = {
            "model_name":
            "TBATS",
            "_series":
            series_id,
            "test_predictions":
            list(pred),
            "test_confidence_intervals":
            convert_confs(conf),
            "mean_absolute_percent_error":
            metrics[0]["mean_absolute_percent_error"],
            "train_test_split_index":
            ts.train_test_split
        }

    elif model == "ETS":
        best_model, pred, conf, metrics = models.train_expsmoothing(ts)

        d = {
            "model_name":
            "ETS",
            "_series":
            series_id,
            "test_predictions":
            list(pred),
            "test_confidence_intervals":
            convert_confs(conf),
            "mean_absolute_percent_error":
            metrics[0]["mean_absolute_percent_error"],
            "train_test_split_index":
            ts.train_test_split
        }

    return d


def calculate_statistics(series_id, experiment_id):
    """Returns a stax.TimeSeries"""
    series_res = requests.get(f"{STAX_BACKEND_API}/api/series/{series_id}",
                              headers=HEADERS)

    experiment_res = requests.get(
        f"{STAX_BACKEND_API}/api/experiments/{experiment_id}", headers=HEADERS)

    experiment = json.loads(experiment_res.content)
    series = json.loads(series_res.content)

    var_name = series["metadata"]["variable_name"]
    df = series_to_df(series).set_index("Date")[var_name]
    ts = TimeSeries(df, series["metadata"]["frequency"], 0.8)
    ts.calculate_statistics()
    return ts


def post_model(data):
    """Returns a requests.Response"""
    post_request = requests.post(f"{STAX_BACKEND_API}/api/models",
                                 headers=HEADERS,
                                 data=simplejson.dumps(data))
    return post_request


def post_decomp(ts, series_id, experiment_id):
    """Returns a requests.Response"""
    decomp = ts.experiment_results["seasonal_decomposition"]
    decomp["_series"] = series_id
    decomp["method"] = "multiplicative"

    decomp_response = requests.post(f"{STAX_BACKEND_API}/api/decompositions",
                                    headers=HEADERS,
                                    data=simplejson.dumps(decomp,
                                                          ignore_nan=True))
    return decomp_response


def post_autocorr(ts, series_id, experiment_id):
    """Returns a requests.Response"""
    autocorr = ts.experiment_results["autocorrelation"]
    autocorr["_series"] = series_id
    autocorr_response = requests.post(
        f"{STAX_BACKEND_API}/api/autocorrelations",
        headers=HEADERS,
        data=simplejson.dumps(autocorr, ignore_nan=True))
    return autocorr_response


def run_arima_job(series_id, experiment_id):
    print(f"Running ARIMA Job at {datetime.datetime.now()}")
    data = train_model(series_id, experiment_id, "ARIMA")
    r = post_model(data)
    response_data = json.loads(r.content)
    print(
        f"ARIMA Job Complete at {datetime.datetime.now()} with status code {r.status_code}"
    )
    # Update experiments
    print("Updating Experiments Data")
    experiment_data = get_experiment(experiment_id)
    experiment_data["_models"].append(response_data["_id"])
    res = put_experiment(experiment_id, experiment_data)
    print(f"Experiment Update compete with status code {res}")
    assert res.status_code == 200
    return {"model_response": r}


def run_tbats_job(series_id, experiment_id):
    """ Returns a dict"""
    print(f"Running TBATS Job at {datetime.datetime.now()}")

    data = train_model(series_id, experiment_id, "TBATS")
    print("TBATS DATA:")
    print(data)
    r = post_model(data)
    response_data = json.loads(r.content)

    # Update experiments
    print("Updating Experiments Data")
    experiment_data = get_experiment(experiment_id)
    experiment_data["_models"].append(response_data["_id"])
    res = put_experiment(experiment_id, experiment_data)
    print(f"Experiment Update compete with status code {res}")
    assert res.status_code == 200

    print(
        f"TBATS Job Complete at {datetime.datetime.now()} with status code {r.status_code}"
    )
    return {"model_response": r}


def run_ets_job(series_id, experiment_id):
    """ Returns a dict"""
    print(f"Running ETS Job at {datetime.datetime.now()}")
    data = train_model(series_id, experiment_id, "ETS")
    r = post_model(data)
    response_data = json.loads(r.content)
    # Update experiments
    print("Updating Experiments Data")
    experiment_data = get_experiment(experiment_id)
    experiment_data["_models"].append(response_data["_id"])
    res = put_experiment(experiment_id, experiment_data)
    print(f"Experiment Update compete with status code {res}")
    assert res.status_code == 200

    print(
        f"ETS Job Complete at {datetime.datetime.now()} with status code {r.status_code}"
    )
    return {"model_response": r}


def run_statistics_job(series_id, experiment_id):
    """ Returns a dict"""
    print(f"Running Statistics Job at {datetime.datetime.now()}")
    ts = calculate_statistics(series_id, experiment_id)

    decomp_response = post_decomp(ts, series_id, experiment_id)
    autocorr_response = post_autocorr(ts, series_id, experiment_id)

    decomp_data = json.loads(decomp_response.content)
    autocorr_data = json.loads(autocorr_response.content)

    decomp_id = decomp_data["_id"]
    autocorr_id = decomp_data["_id"]

    print(
        f"Statistics Job Complete at {datetime.datetime.now()} with status code {decomp_response.status_code} and {autocorr_response.status_code}"
    )

    # Update experiments
    print("Updating Experiments Data")
    experiment_data = get_experiment(experiment_id)
    experiment_data["_decomposition"] = decomp_id
    experiment_data["_autocorrelation"] = autocorr_id
    res = put_experiment(experiment_id, experiment_data)
    print(f"Experiment Update compete with status code {res}")
    assert res.status_code == 200

    return {
        "decomp_response": decomp_response,
        "autocorr_response": autocorr_response
    }
