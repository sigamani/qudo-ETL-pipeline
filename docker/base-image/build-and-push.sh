docker build -t europe-west2-docker.pkg.dev/clustered-cream/public-images/python-with-r:latest "`dirname $0`"/.
docker push europe-west2-docker.pkg.dev/clustered-cream/public-images/python-with-r:latest