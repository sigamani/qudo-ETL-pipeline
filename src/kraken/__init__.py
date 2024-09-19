import Poseidon
from dagster import Definitions, load_assets_from_modules
from dagster_aws.s3 import S3Resource

from kraken import assets
from kraken.app import aws_init, main

all_assets = load_assets_from_modules([assets, aws_init, main, Poseidon.assets])

defs = Definitions(
    assets=all_assets,
    resources={'s3': S3Resource(region_name='eu-west-2')}
)
