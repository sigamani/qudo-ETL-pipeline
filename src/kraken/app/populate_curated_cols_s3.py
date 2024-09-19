import boto3
import json
import s3fs
import pandas as pd

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
s3_check = s3fs.S3FileSystem()

bucket = 'qudo-datascience'

sp_tag = False

survey_name = "qudo_consumerzeitgeist_uk_q2_2023_test"
trunc_surv_name = "survey_name_with_no_tags"
environ = "test"
filename_metadata = "s3://qudo-datascience/data-store/lachsesis/test/qudo_zeitgeist/zeitgeist_curated_cols.csv"

metadata = pd.read_csv(filename_metadata)

if "industry" in metadata.columns:

    for industry, segmentation, col_list in zip(metadata.industry, metadata.segmentation, metadata.cols):
        try:
            col_list = col_list.split(",")
        except AttributeError:
            pass
        if sp_tag:
            key = f'data-store/lachsesis/{environ}/{survey_name.split("_")[0]}/{trunc_surv_name}/{industry}/curated/{segmentation}/cols.json'
        else:
            key = f'data-store/lachsesis/{environ}/{survey_name}/{industry}/curated/{segmentation}/cols.json'
        s3_client.put_object(Body=json.dumps(col_list), Bucket=bucket, Key=key)

else:
    for segmentation, col_list in zip(metadata.segmentation, metadata.cols):
        try:
            col_list = col_list.split(",")
        except AttributeError:
            pass

        if sp_tag:
            key = f'data-store/lachsesis/{environ}/{survey_name.split("_")[0]}/{trunc_surv_name}/curated/{segmentation}/cols.json'
        else:
            key = f'data-store/lachsesis/{environ}/{survey_name}/curated/{segmentation}/cols.json'
        s3_client.put_object(Body=json.dumps(col_list), Bucket=bucket, Key=key)