FROM python:3.8-buster

RUN apt-get update && \
    apt-get install -y gcc make apt-transport-https ca-certificates build-essential  

WORKDIR  /usr/presence

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /src/

RUN cd src/app

CMD ["python", "app.py"]


