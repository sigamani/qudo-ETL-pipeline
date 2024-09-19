docker run \
  --env-file "`dirname $0`"/.env \
  --publish 8080:3000 \
  europe-west2-docker.pkg.dev/clustered-cream/ak-dagster/poc:latest