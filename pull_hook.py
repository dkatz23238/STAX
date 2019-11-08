import json
import os
import time

import requests
from redis import Redis
from rq import Queue

from server_utils import run_primary_job
import pymongo

MONGO_DB_URI = os.environ.get("MONGO_DB_URI")

client = pymongo.MongoClient(MONGO_DB_URI)
db = client.get_database()

experiments = db["experiments"]
tokens = db["tokens"]
enqueued_experiments = db["enqueued_experiments"]

STAX_BACKEND_API = os.environ.get("STAX_BACKEND_URI")

HEADERS = {
    "X-Auth-Token": os.environ.get("BACKEND_AUTH_TOKEN"),
    "content-type": "application/json"
}

REDIS_HOST = os.environ.get("REDIS_HOST")

redis_conn = Redis(host=REDIS_HOST)
q = Queue(connection=redis_conn,
          default_timeout=600)  # no args implies the default queue


def get_jobs(q):
    jobs = [{
        "job": job.id,
        "enqueued_at": job.enqueued_at.isoformat()
    } for job in q.get_jobs()]
    return jobs


while True:
    time.sleep(1)

    for pending_experiments in experiments.find({"status": "pending"}):
        # Post Job To Queue
        _series = pending_experiments["_series"]
        _experiment = pending_experiments["_id"]

        userUID = pending_experiments["userUID"]
        user_token = tokens.find_one({"userUID": userUID})["token"]

        experiment_check = list(
            enqueued_experiments.find({"_experiment": _experiment}))

        if len(experiment_check) > 0:
            pass

        else:
            r = requests.get(f"{STAX_BACKEND_API}/api/series/{_series}",
                             headers={"X-Auth-Token": user_token})
            print(r)
            job = q.enqueue(run_primary_job,
                            args=(r, _experiment, user_token),
                            timeout=600)
            print("Job Enqued!")
            res = {
                "job_enqueued_at": job.enqueued_at.isoformat(),
                "job_status": job.get_status(),
                "job": job.id,
            }
            print(res)

            enqueued_experiments.insert_one({"_experiment": _experiment})
