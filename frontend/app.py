from flask import Flask, render_template
from flask import Markup
import numpy as np
import json
import glob
import datetime
import uuid

def make_uuid():
    return str(uuid.uuid4())[0:8]

app = Flask(__name__)

@app.route('/home')
def index():
    return render_template("index.html")  

@app.route('/jobs')
def jobs():
    jobs = 12
    njobs = str(jobs)
    job_data = [
        {"title": make_uuid()}
        for i in range(jobs)
    ]
    return render_template("jobs.html", njobs=njobs, job_data=job_data)  

@app.route('/results')
def results():
    return render_template("results.html")  

@app.route('/datasets')
def datasets():
    return render_template("datasets.html")  


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
