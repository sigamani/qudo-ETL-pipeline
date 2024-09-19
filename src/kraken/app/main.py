import datetime
import json
import pickle
from itertools import chain

import boto3
import pandas as pd
from dagster import asset

from kraken.app.SegmentationConfig import SegmentationConfig
from kraken.app.clustering import Clusterings
from kraken.app.s3_data_fetching_functions import fetch_and_generate_clustering_cols_dict_s3, fetch_ml_columns_s3
from kraken.app.utils import remove_time_cols, append_manual_seg_columns, add_tgt_tag, remove_numeric_tag

bucket = 'qudo-datascience'


def map_to_option_title(survey_name, r_responses_df, environ, sp_tag):
    if sp_tag:
        trunc_surv_name = "_".join(survey_name.split("_")[0:-2])
        preprocessed_questions_uri = f's3://{bucket}/data-store/special_projects/{environ}/{survey_name.split("_")[0]}/{trunc_surv_name}/{trunc_surv_name}_questions_preprocessed.parquet'
    else:
        preprocessed_questions_uri = f's3://qudo-datascience/data-store/surveys/{environ}/questions_preprocessed/{survey_name}/{survey_name}.parquet'
    preprocessed_questions_df = pd.read_parquet(preprocessed_questions_uri)
    mismatch_df = preprocessed_questions_df.loc[
        ~(preprocessed_questions_df['option_text'] == preprocessed_questions_df['option_value'])][
        ['varname', 'option_text', 'option_value']].drop_duplicates().copy()
    mismatch_questions = mismatch_df['varname'].unique()

    responses_df = r_responses_df.copy()
    responses_columns = responses_df.columns

    for mismatch_col in responses_columns:
        if mismatch_col in mismatch_questions:
            mismatch_mappings = mismatch_df[mismatch_df['varname'] == mismatch_col].copy()
            mismatch_mappings_dict = dict(zip(mismatch_mappings['option_value'], mismatch_mappings['option_text']))
            try:
                responses_df[mismatch_col] = responses_df[mismatch_col].apply(
                    lambda x: mismatch_mappings_dict[x] if x in mismatch_mappings_dict.keys() else x)
            except:
                raise KeyError(f'Mapping is not complete, please check all options for varname {mismatch_col}.')
    return responses_df


@asset()
def do_segmentation_and_save_to_s3(segmentation_config: SegmentationConfig) -> str:
    survey_name = segmentation_config.survey_name
    trunc_surv_name = segmentation_config.trunc_survey_name
    sp_tag = segmentation_config.sp_tag
    environ = segmentation_config.environ
    data_uri = segmentation_config.data_uri
    columns_uri = segmentation_config.columns_uri
    preprocessed = segmentation_config.preprocessed
    weight_column = segmentation_config.weight_column
    data_provided = segmentation_config.data_provided
    hierarchical = segmentation_config.hierarchical
    ignore_hierarchical_value = segmentation_config.ignore_hierarchical_value
    add_manual_seg_columns = segmentation_config.add_manual_seg_columns

    nowish = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S")

    s3 = boto3.resource('s3')

    # preprocessed dataset from ingestion module available
    if preprocessed:
        preprocessed_df = pd.read_parquet(preprocessed)
        # HOTFIX: remove _time_ columns
        preprocessed_df = remove_time_cols(df=preprocessed_df)

    # todo: do we still need this if-else, can it just call on the parquet?
    if data_provided:
        df = data_uri
    else:
        df = pd.read_parquet(data_uri)

    # HOTFIX: remove _time_ columns
    df = remove_time_cols(df=df)

    try:
        df = map_to_option_title(survey_name, df, environ, sp_tag)
    except FileNotFoundError:
        pass

    segmentation_types, col_list = fetch_and_generate_clustering_cols_dict_s3(survey_name=survey_name,
                                                                              trunc_surv_name=trunc_surv_name,
                                                                              environ=environ,
                                                                              sp_tag=sp_tag)

    # todo: this is only used if a dataframe is passed to (could be neglible now)
    if isinstance(add_manual_seg_columns, pd.DataFrame):
        df = append_manual_seg_columns(df=df,
                                       manual_seg_col_df=add_manual_seg_columns)

    ### Column Selection todo: add it to the s3 functions
    if columns_uri and not data_provided:
        splits = columns_uri.split('/')
        bucket_name = splits[2]
        key_name = "/".join(splits[3:])
        content_obj = s3.Object(bucket_name, key_name)
        content = content_obj.get()['Body'].read().decode('utf-8')
        cols = json.loads(content)
    elif data_provided:
        col_fragments = [x.lower() for x in columns_uri]
        cols = []
        for col in col_fragments:
            cols.append([x for x in df.columns if col in x])
        cols = list(chain(*cols))
        cols_json = json.dumps(cols)
        if sp_tag:
            cols_s3_uri = f'data-store/lachsesis/{environ}/{survey_name}/{survey_name.split("_")[0]}/{trunc_surv_name}/data_provided/{nowish}/cols.json'
        else:
            cols_s3_uri = f'data-store/lachsesis/{environ}/{survey_name}/data_provided/{nowish}/cols.json'
        s3.Object(bucket, cols_s3_uri).put(Body=cols_json)
    else:
        cols = fetch_ml_columns_s3(survey_name=survey_name,
                                   trunc_surv_name=trunc_surv_name,
                                   sp_tag=sp_tag,
                                   bucket=bucket,
                                   environ=environ)
    if cols:
        segmentation_types['ml'] = cols
    df.columns = add_tgt_tag(column_list=df.columns.tolist())

    if preprocessed:
        preprocessed_df.columns = remove_numeric_tag(column_list=list(preprocessed_df.columns))
        preprocessed_df.columns = add_tgt_tag(column_list=list(preprocessed_df.columns))

    for segmentation, columns in segmentation_types.items():
        columns = [x.lower() for x in columns]

        if len([x for x in columns if '_tgt' in x]) == 0:
            cols = add_tgt_tag(column_list=columns)
            cols = [x for x in df.columns for col in cols if x == col]

            if preprocessed and segmentation == 'ml':
                cols = remove_numeric_tag(column_list=cols)
        else:
            cols = [x for x in df.columns for col in columns if col in x]

        print(f'*************************** STARTING {segmentation} ***************************')
        if preprocessed and segmentation == 'ml':
            segmentation_obj = Clusterings(survey_name=survey_name,
                                           data=preprocessed_df,
                                           cluster_vars=cols,
                                           hierarchical=hierarchical,
                                           ignore_hierarchical_value=ignore_hierarchical_value,
                                           full_data=df,
                                           weight_col=weight_column)
        else:
            segmentation_obj = Clusterings(survey_name=survey_name,
                                           data=df,
                                           cluster_vars=cols,
                                           hierarchical=hierarchical,
                                           ignore_hierarchical_value=ignore_hierarchical_value,
                                           weight_col=weight_column)

        rules_based = [x for x in df.columns if segmentation in x.lower() and 'qudo' in x.lower()]
        segmentations = segmentation_obj.run_all_segmentations(q_code=rules_based)

        metrics_list = []
        for x in segmentations.values():
            if isinstance(x, list):
                for y in x:
                    metrics_list.append(y['metrics'])
            else:
                metrics_list.append(x['metrics'])

        metrics_df = pd.DataFrame(x for x in metrics_list)

        if sp_tag:
            metrics_s3_uri = f's3://qudo-datascience/data-store/kraken_outputs/{environ}/{survey_name.split("_")[0]}/{trunc_surv_name}/{segmentation}/metrics.csv'
        else:
            metrics_s3_uri = f's3://qudo-datascience/data-store/kraken_outputs/{environ}/{survey_name}/{segmentation}/metrics.csv'
        metrics_df.to_csv(metrics_s3_uri, index=False)

        for seg_name, seg_data in segmentations.items():
            if sp_tag:
                seg_data_s3_uri = f'data-store/kraken_outputs/{environ}/{survey_name.split("_")[0]}/{trunc_surv_name}/{segmentation}/{seg_name}.pickle'
            else:
                seg_data_s3_uri = f'data-store/kraken_outputs/{environ}/{survey_name}/{segmentation}/{seg_name}.pickle'
            pickle_byte_obj = pickle.dumps(seg_data)
            s3.Object('qudo-datascience', seg_data_s3_uri).put(Body=pickle_byte_obj)
    return f'data-store/kraken_outputs/{environ}/{survey_name}/{nowish}'
