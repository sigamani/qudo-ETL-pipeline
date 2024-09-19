FROM python:3.11-slim
RUN apt-get update && apt-get install -y r-base

RUN mkdir -p /opt/dagster/app

WORKDIR /opt/dagster/app

COPY setup.py /opt/dagster/app/
RUN pip install .