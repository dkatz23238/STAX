import json
import os
import time
import datetime

import requests
from redis import Redis
from rq import Queue

from stax.microservices import run_arima_job, run_ets_job, run_statistics_job, run_tbats_job
import pymongo

MONGO_DB_URI = os.environ.get("MONGO_DB_URI")
STAX_BACKEND_API = os.environ.get("STAX_BACKEND_URI")

HEADERS = {
    "X-Auth-Token": os.environ.get("BACKEND_AUTH_TOKEN"),
    "content-type": "application/json"
}

REDIS_HOST = os.environ.get("REDIS_HOST")

# Mongo stuff
client = pymongo.MongoClient(MONGO_DB_URI)
db = client.get_database()

experiments = db["experiments"]
tokens = db["tokens"]
enqueued_experiments = db["enqueued_experiments"]

# Redis Stuff

redis_conn = Redis(host=REDIS_HOST)
q = Queue(connection=redis_conn,
          default_timeout=1200)  # no args implies the default queue

while True:
    for pending_experiments in experiments.find({"status": "pending"}):
        # Post Job To Queue
        _series = pending_experiments["_series"]
        _experiment = pending_experiments["_id"]

        userUID = pending_experiments["userUID"]
        user_token = tokens.find_one({"userUID": userUID})["token"]

        # Check if the experiment is not already enqueued
        experiment_check = list(
            enqueued_experiments.find({"_experiment": str(_experiment)}))

        # If the list is empty then enqueue the job
        if len(experiment_check) == 0:
            print(f"Enqueueing Jobs on {_experiment}")

            # Jobs to queue
            jobs_to_do = [
                run_arima_job, run_ets_job, run_statistics_job, run_tbats_job
            ]

            enqueued_at = datetime.datetime.utcnow()

            for task in jobs_to_do:
                # Enqueue to redis queue for worker to pick up
                print(f"Enqueuing Task {task}")
                job = q.enqueue(task,
                                args=(str(_series), str(_experiment),
                                      str(user_token)),
                                timeout=1200)

            # Make sure the series is now in the enqueued collection.
            enqueued_experiments.insert_one(
                {"_experiment": str(_experiment), "enqueued_at": enqueued_at})
            print("Enqueued Experiment Sent to DB")

        time.sleep(20)
