import os
from datetime import datetime, timedelta
from flask import Flask, request, abort, jsonify
import requests
from redis import Redis
from rq import Queue
import json
from server_utils import run_primary_job

app = Flask(__name__)
STAX_BACKEND_API = STAX_BACKEND_API = os.environ.get("STAX_BACKEND_URI")
HEADERS = {
    "X-Auth-Token": os.environ.get("BACKEND_AUTH_TOKEN"),
    "content-type": "application/json"
}
REDIS_HOST = os.environ.get("REDIS_HOST")

redis_conn = Redis(host=REDIS_HOST)
q = Queue(connection=redis_conn,
          default_timeout=600)  # no args implies the default queue


def validate_json(data):
    """ Checks if keys are in json. """
    if "_series" in data.keys() and "_experiment" in data.keys():
        return True
    else:
        return False


def check_resources_exists(data):
    """ returns True if both resources exist, else resturns false. """
    series_id = data["_series"]
    experiment_id = data["_experiment"]
    series_r = requests.get(f"{STAX_BACKEND_API}/api/series/{series_id}",
                            headers=HEADERS)
    experiment_r = requests.get(
        f"{STAX_BACKEND_API}/api/experiments/{experiment_id}", headers=HEADERS)

    if series_r.status_code != 200 or experiment_r.status_code != 200:
        return False
    else:
        return True


def get_jobs(q):
    jobs = [{
        "job": job.id,
        "enqueued_at": job.enqueued_at.isoformat()
    } for job in q.get_jobs()]
    return jobs


@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    if request.method == 'GET':
        jobs = get_jobs(q)
        return jsonify(jobs), 200

    # Post request to trigger job
    elif request.method == 'POST':

        # Check if JSON
        if not request.is_json:
            abort(400)

        #Check JSON Schema valid
        if not validate_json(request.json):
            abort(400)

        # Check Series Exist
        if not check_resources_exists(request.json):
            abort(400)

        # Post Job To Queue
        _series = request.json["_series"]
        _experiment = request.json["_experiment"]
        r = requests.get(f"{STAX_BACKEND_API}/api/series/{_series}",
                         headers=HEADERS)
        print(r)
        job = q.enqueue(run_primary_job, args=(r, _experiment), timeout=600)
        print("Job Enqued!")
        res = {
            "job_enqueued_at": job.enqueued_at.isoformat(),
            "job_status": job.get_status(),
            "job": job.id,
        }

        # return Job ID
        return jsonify(res), 202

    else:
        abort(400)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000")