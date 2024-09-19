import json
import os
from io import StringIO

import boto3
import s3fs
from dagster import Any, asset, Optional

from kraken.app.SegmentationConfig import SegmentationConfig
from kraken.app.main import do_segmentation_and_save_to_s3
from kraken.app.s3_data_fetching_functions import check_essential_columns_s3, fetch_essential_columns_s3, \
    fetch_ml_columns_uri
from kraken.app.utils import find_weight_col

pipeline_env = os.getenv('PIPELINE_ENV', 'test')
bucket = 'qudo-datascience'
collected_surveys_key = f"data-store/codebuild-resources/{pipeline_env}/processed_surveys_log/collected_surveys.json"


@asset()
def segmentation_config(next_survey: dict[str, Any]) -> Optional[SegmentationConfig]:
    s3_check = s3fs.S3FileSystem()

    survey_name = next_survey['title']
    trunc_surv_name = "_".join(survey_name.split("_")[0:-2])

    sp_tag = False
    if "_sp_" in next_survey['title'].lower():
        sp_tag = True

    if sp_tag:
        file = f's3://{bucket}/data-store/special_projects/{pipeline_env}/{survey_name.split("_")[0]}/{trunc_surv_name}/{trunc_surv_name}_responses.parquet'
        PROJECT_NAME = trunc_surv_name
    else:
        file = f's3://{bucket}/data-store/surveys/{pipeline_env}/responses_cleaned/{survey_name}/{survey_name}.parquet'
        PROJECT_NAME = "_".join(next_survey['title'].split("_")[:-1])

    if check_essential_columns_s3(project_name=PROJECT_NAME):
        essential_columns = fetch_essential_columns_s3(bucket=bucket, project_name=PROJECT_NAME)
        if not essential_columns['execute']['platform']:
            return None
        WEIGHT_COLUMN = find_weight_col(data=file,
                                        essential_columns=essential_columns)
    else:
        WEIGHT_COLUMN = None

    cols_uri = fetch_ml_columns_uri(survey_name=survey_name,
                                    trunc_surv_name=trunc_surv_name,
                                    sp_tag=sp_tag,
                                    bucket=bucket,
                                    pipeline_env=pipeline_env)
    if sp_tag:
        preprocessed_file = f's3://{bucket}/data-store/special_projects/{pipeline_env}/{survey_name.split("_")[0]}/{trunc_surv_name}/{trunc_surv_name}_responses_preprocessed_cols_to_drop.parquet'
    else:
        preprocessed_file = f's3://{bucket}/data-store/surveys/{pipeline_env}/responses_preprocessed/{survey_name}/{survey_name}.parquet'
    preprocessed_file_exists = s3_check.exists(preprocessed_file)

    return SegmentationConfig(survey_name=next_survey['title'],
                              trunc_survey_name=trunc_surv_name,
                              sp_tag=sp_tag,
                              data_uri=file,
                              columns_uri=cols_uri,
                              preprocessed=preprocessed_file if preprocessed_file_exists else None,
                              environ=pipeline_env,
                              weight_column=WEIGHT_COLUMN)


@asset()
def process_segmentation(collected_surveys: list[dict[str, Any]],
                         next_survey: dict[str, Any],
                         do_segmentation_and_save_to_s3: str) -> None:
    session = boto3.Session()
    s3 = session.resource('s3')
    processed_survey = next(survey for survey in collected_surveys if survey["id"] == next_survey["id"])
    processed_survey["processed_by"].append("kraken")
    json_buffer = StringIO()
    json.dump(collected_surveys, json_buffer, indent=2)
    s3.Object(bucket, collected_surveys_key).put(Body=json_buffer.getvalue())
