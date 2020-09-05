FROM tiangolo/uwsgi-nginx-flask:latest
LABEL maintainer="ricardo.alvarezz@usach.cl"

COPY ./app /app
WORKDIR /app

RUN apt-get update \
&& apt-get install -y --no-install-recommends gcc build-essential \
&& pip install --upgrade pip \
&& pip install -r requirement.txt