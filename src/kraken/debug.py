import Poseidon
from dagster import materialize, load_assets_from_modules
from dagster_aws.s3 import S3Resource

from kraken import assets
from kraken.app import aws_init, main

# run this in Debug mode to step in
if __name__ == "__main__":
    materialize(
        assets=load_assets_from_modules([Poseidon.assets, aws_init, main, assets]),
        resources={'s3': S3Resource(region_name='eu-west-2')}
    )
