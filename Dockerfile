FROM python:3.7-stretch

RUN apt-get update

WORKDIR /root
COPY ./ ./
RUN pip install -qr ./requirements.txt

ENTRYPOINT [ "python", "-u", "app.py" ] 
