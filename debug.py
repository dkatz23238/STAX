import os, json, time

import uuid
import requests
import time
from stax import TimeSeries
import pandas as pd
import datetime


def check_series():
    r = requests.get("http://localhost:81/api/series?token=6c11bffd")

    assert r.status_code == 200

    data = json.loads(r.content)

    return data


def check_experiments():
    r = requests.get("http://localhost:81/api/experiments?token=6c11bffd")

    assert r.status_code == 200

    data = json.loads(r.content)

    return data


def get_series(_id):
    series = check_series()
    for s in series:
        if s["_id"] == _id:
            return s


def series_to_df(data):
    df = pd.DataFrame(data["data"])
    df.columns = ["Date", data["metadata"]["variable_name"].capitalize()]
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def run_analysis(df, column, frequency):
    if not "Date" in df.columns:
        raise Exception("No Date")

    if not column in df.columns:
        raise Exception("No Date")

    series = df.set_index("Date")[column]
    ts = TimeSeries(series, frequency, 0.8)
    ts.calculate_statistcs()
    # Train and select models
    ts.train_models()
    return ts


def convert_experiment_to_dict(data, id):
    for k in data["models"]:
        data["models"][k]["model"] = str(data["models"][k]["model"])
    data["_id"] = id
    return data


def conduct_db_check():
    '''Retruns True if there are less experiments than time series in the database, False if there are none. '''
    series_check = check_series()
    exp_check = check_experiments()

    return len(exp_check) < len(series_check)


def get_pending_experiment_ids():
    series_check = check_series()
    exp_check = check_experiments()
    experiments_ids = [i["_id"] for i in exp_check]
    pending_analysis = [
        i["_id"] for i in series_check if i["_id"] not in experiments_ids
    ]
    return pending_analysis


def insert_experiment_to_db(data):
    j = json.dumps(data)
    r = requests.post("http://localhost:81/api/experiments?token=6c11bffd",
                      data=j)
    return r


while True:
    print("Conducting insert check at %s" %
          datetime.datetime.now().isoformat())
    db_check = conduct_db_check()
    pending = get_pending_experiment_ids()
    print('Database check: %s' % db_check)
    print('Pending series to process: %s' % pending)

    if db_check:
        print("Conducting inserts at %s" % datetime.datetime.now().isoformat())
        for _id in pending:
            # Get the pending series from backend
            data = get_series(_id)
            # Variable name
            v_name = data["metadata"]["variable_name"].capitalize()
            # Time series frequency
            freq = data["metadata"]["frequency"]
            # Convert to pandas.DataFrame
            df = series_to_df(data)
            # Convert to stax.TimeSeries
            ts = run_analysis(df, v_name, freq)
            # Get the result of analysis
            result = convert_experiment_to_dict(ts.experiment_results, id=_id)
            # Database insertion
            r = insert_experiment_to_db(result)
            assert r.status_code == 202
            print(r)
            print(_id)
    time.sleep(30)