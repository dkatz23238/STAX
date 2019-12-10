FROM python:3.6-slim-stretch

RUN apt-get update

WORKDIR /root
COPY ./ ./
RUN pip install -qr ./requirements.txt

# RUN ["python", "test.py"]
# ENV BACKEND_URL="https://stax-backend.crossentropy.solutions"
EXPOSE 5000
ENTRYPOINT [ "python", "-u","-m", "stax.microservices.experiment_enqueuer" ] 
