import json
import os
import time

import requests
from redis import Redis
from rq import Queue

from stax.microservices import run_arima_job, run_ets_job, run_statistics_job, run_tbats_job, get_experiment, put_experiment
import pymongo

MONGO_DB_URI = os.environ.get("MONGO_DB_URI")
STAX_BACKEND_API = os.environ.get("STAX_BACKEND_URI")
TOKEN = ""

REDIS_HOST = os.environ.get("REDIS_HOST")


if __name__ == "__main__":
    # Mongo stuff
    client = pymongo.MongoClient(MONGO_DB_URI)
    db = client.get_database()

    # Mongo Collections
    experiments = db["experiments"]
    tokens = db["tokens"]
    enqueued_experiments = db["enqueued_experiments"]

    while True:
        # Find pending experiments
        query = experiments.find({"status": "pending"})
        for experiment in query:
            try:
                # Has the experiment completed successfully?
                if ((len(experiment["_models"]) == 3) &
                    (experiment["_decomposition"] is not None) &
                        (experiment["_autocorrelation"] is not None)):

                    _experiment = experiment["_id"]

                    userUID = experiment["userUID"]
                    user_token = tokens.find_one({"userUID": userUID})["token"]

                    print("Updating Experiments Data")
                    # The experiment is now complete
                    experiment_data = {}
                    experiment_data["status"] = "complete"
                    # Update in the backend with PUT request
                    res = put_experiment(experiment["_id"], experiment_data,
                                         user_token)
                    print(f"Experiment Update compete with status code {res}")
                    assert res.status_code == 200, "Server response was not 200 for PUT request."
            except Exception as e:
                print(f"Exception Handling: {e}")

        # Wait and repeat
        time.sleep(5)
