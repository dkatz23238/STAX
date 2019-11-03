import requests

data = {
    "_series": "5dbb431bed05ac1b389e9845",
    "_experiment": "5dbb431bed05ac1b389e9846"
}

for i in range(2):
    r = requests.post("http://localhost:5000/webhook", json=data)