docker build -t europe-west2-docker.pkg.dev/clustered-cream/ak-dagster/poc:latest "`dirname $0`"/../src
docker images | grep "ak-dagster"