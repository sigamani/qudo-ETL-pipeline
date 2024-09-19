import json

from typing import Any, Iterator

from dagster import asset, ResourceParam, AssetExecutionContext, Output, In, DagsterType
from dagster_aws.s3 import S3Resource


@asset
def collected_surveys(s3: ResourceParam[S3Resource]) -> list[dict[str, Any]]:
    bucket = 'qudo-datascience'
    collected_surveys_key = "data-store/codebuild-resources/test/processed_surveys_log/collected_surveys.json"
    file = s3.get_client().get_object(Bucket=bucket, Key=collected_surveys_key)
    return json.load(file['Body'])



@asset(output_required=False)
def next_survey(context: AssetExecutionContext, collected_surveys: list[dict[str, Any]], sample_asset: int) -> Iterator[dict[str, Any]]:
    unprocessed_surveys = [s for s in collected_surveys if "kraken" not in s["processed_by"]]
    context.add_output_metadata({ sample_asset: sample_asset })
    if len(unprocessed_surveys) > 0:
        first_unprocessed_survey = unprocessed_surveys[0]
        context.add_output_metadata(first_unprocessed_survey)
        yield Output(first_unprocessed_survey)
