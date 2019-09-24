FROM python:3.7-stretch

RUN apt-get update

WORKDIR /root
COPY ./ ./
RUN pip install -qr ./requirements.txt

RUN ["python", "test.py"]
# ENV BACKEND_URL="https://stax-backend.crossentropy.solutions"
ENTRYPOINT [ "python", "-u", "app.py" ] 
