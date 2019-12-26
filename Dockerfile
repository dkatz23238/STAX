FROM python:3.6-slim-stretch

RUN apt-get update

WORKDIR /root
COPY ./ ./
RUN pip install -qr ./requirements.txt && python setup.py develop

RUN cd tests && pytest unit_tests.py && cd /root

EXPOSE 5000
ENTRYPOINT [ "python", "-u","-m", "stax.microservices.experiment_enqueuer" ] 

