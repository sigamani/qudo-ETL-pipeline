import boto3
import ast
import json
import s3fs

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
s3_check = s3fs.S3FileSystem()


def check_essential_columns_s3(project_name):
    essential_columns_key = f"data-store/codebuild-resources/essential_columns/essentialcolumns_{project_name}.json"
    s3_check = s3fs.S3FileSystem(use_listings_cache=False)
    if s3_check.exists(f's3://qudo-datascience/{essential_columns_key}'):
        return True
    else:
        return False


def fetch_essential_columns_s3(bucket, project_name):

    essential_columns_key = f"data-store/codebuild-resources/essential_columns/essentialcolumns_{project_name}.json"
    essential_columns_file = s3_client.get_object(Bucket=bucket, Key=essential_columns_key)
    essential_columns = json.load(essential_columns_file['Body'])

    return essential_columns


def fetch_and_generate_clustering_cols_dict_s3(survey_name, trunc_surv_name, environ, sp_tag):
    s3_client = boto3.client('s3')
    s3 = boto3.resource('s3')
    bucket_name = 'qudo-datascience'
    if sp_tag:
        curated_cols_folders = f'data-store/lachsesis/special_projects/{environ}/{survey_name.split("_")[0]}/{trunc_surv_name}/curated/'
        ml_cols_folder = f'data-store/lachsesis/special_projects/{environ}/{survey_name.split("_")[0]}/{trunc_surv_name}/ml/'
    else:
        curated_cols_folders = f'data-store/lachsesis/{environ}/{survey_name}/curated/'
        ml_cols_folder = f'data-store/lachsesis/{environ}/{survey_name}/ml/'

    segmentation_types = {}
    if s3_check.exists(f's3://qudo-datascience/{curated_cols_folders}'):
        curated_cols = s3_client.list_objects(Bucket=bucket_name, Prefix=curated_cols_folders)
        try:
            for filename in curated_cols['Contents']:
                for k, v in filename.items():
                    if k == 'Key':
                        if v[-9:] == 'cols.json':
                            content_obj = s3.Object(bucket_name, v)
                            content = content_obj.get()['Body'].read().decode('utf-8')
                            cols = json.loads(content)
                            if not cols:
                                cols = ast.literal_eval(content)
                            splits = v.split('/')
                            segmentation_name = splits[-2]
                            segmentation_types[segmentation_name] = cols

            ml_cols = s3_client.list_objects(Bucket=bucket_name, Prefix=ml_cols_folder)

            for filename in ml_cols['Contents']:
                cols_list = []
                for k, v in filename.items():
                    if k == 'Key':
                        if v[-9:] == 'cols.json':
                            cols_list.append(v)
        except KeyError:
            cols_list = []
    else:
        cols_list = []

    return segmentation_types, cols_list


def fetch_ml_columns_s3(survey_name, trunc_surv_name, sp_tag, bucket, environ):
    if sp_tag:
        ml_cols_json = f'data-store/lachsesis/{environ}/{survey_name.split("_")[0]}/{trunc_surv_name}/ml/cols.json'
    else:
        ml_cols_json = f'data-store/lachsesis/{environ}/{survey_name}/ml/cols.json'
    if s3_check.exists(f's3://{bucket}/{ml_cols_json}'):
        try:
            file = s3_client.get_object(Bucket=bucket, Key=ml_cols_json)
            cols = json.load(file['Body'])
        except:
            file = s3_client.get_object(Bucket=bucket, Key=ml_cols_json + 'cols.json')
            cols = json.load(file['Body'])
    else:
        return None

    return cols


def fetch_ml_columns_uri(survey_name, trunc_surv_name, sp_tag, bucket, pipeline_env):
    if sp_tag:
        ml_cols_json = f'data-store/lachsesis/{pipeline_env}/{survey_name.split("_")[0]}/{trunc_surv_name}/ml/cols.json'
    else:
        ml_cols_json = f'data-store/lachsesis/{pipeline_env}/{survey_name}/ml/cols.json'
    if s3_check.exists(f's3://{bucket}/{ml_cols_json}'):
        cols_uri = f's3://{bucket}/{ml_cols_json}'
    else:
        cols_uri = None
        print(f"ml cols uri does NOT exist in this s3 location: [{ml_cols_json}]")

    return cols_uri
