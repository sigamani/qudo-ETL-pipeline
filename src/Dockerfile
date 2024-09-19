FROM europe-west2-docker.pkg.dev/clustered-cream/public-images/python-with-r

RUN mkdir -p /opt/dagster/dagster_home /opt/dagster/app
ENV DAGSTER_HOME=/opt/dagster/dagster_home/
COPY kraken/deploy/dagster.yaml /opt/dagster/dagster_home/

WORKDIR /opt/dagster/app

COPY setup.py /opt/dagster/app/
RUN pip install .

COPY pyproject.toml /opt/dagster/app/
COPY kraken/ /opt/dagster/app/kraken/

EXPOSE 3000

ENTRYPOINT ["dagster-webserver", "-h", "0.0.0.0", "-p", "3000"]
