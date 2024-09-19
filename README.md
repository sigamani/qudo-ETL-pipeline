## What is this repository for?

This is a [Dagster](https://dagster.io/) project scaffolded with [`dagster project scaffold`](https://docs.dagster.io/getting-started/create-new-project).

## Authenticating against the Kluster python package registry

See: https://cloud.google.com/artifact-registry/docs/python/authentication

View the registry in GCP here: 
https://console.cloud.google.com/artifacts/docker/klusterai-dev/us-east4/python-packages-v2?project=klusterai-dev

```shell
gcloud artifacts print-settings python \
    --project=klusterai-dev \
    --repository=python-packages-v2 \
    --location=us-east4
```

```
# Insert the following snippet into your .pypirc

[distutils]
index-servers =
    python-packages-v2

[python-packages-v2]
repository: https://us-east4-python.pkg.dev/klusterai-dev/python-packages-v2/

# Insert the following snippet into your pip.conf

[global]
extra-index-url = https://us-east4-python.pkg.dev/klusterai-dev/python-packages-v2/simple/
```

## Getting started

First, install your Dagster code location as a Python package. By using the --editable flag, pip will install your Python package in ["editable mode"](https://pip.pypa.io/en/latest/topics/local-project-installs/#editable-installs) so that as you develop, local code changes will automatically apply.

```bash
cd src
pip install -e ".[dev]"
```

Then, start the Dagster UI web server:

```bash
cd src
dagster dev
```

Open http://localhost:3000 with your browser to see the project.

You can start writing assets in `src/kraken/assets.py`. The assets are automatically loaded into the Dagster code location as you define them.

## Development


### Adding new Python dependencies

You can specify new Python dependencies in `src/setup.py`.

### Unit testing

Tests are in the `src/kraken_tests` directory and you can run tests using `pytest`:

```bash
pytest
```
