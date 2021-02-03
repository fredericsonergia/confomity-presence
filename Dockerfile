FROM python:3.8-buster

RUN apt-get update && \
    apt-get install -y gcc make apt-transport-https ca-certificates build-essential
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR  /usr/src/presence

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY src src

WORKDIR /usr/src/presence/src/app





