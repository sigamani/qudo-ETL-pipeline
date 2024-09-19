import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


def find_weight_col(data, essential_columns) -> str:
    if (not essential_columns["weighting"]["utility"]["pre_completes"]) and (
    not essential_columns["weighting"]["utility"]["post_completes"]):
        print("Weight column not in dataset - weighting might have not been performed")
        weight_col = None
        return weight_col

    elif essential_columns["weighting"]["utility"]["pre_completes"] and essential_columns["weighting"]["utility"][
        "post_completes"]:
        weight_col = "weight"

    elif essential_columns["weighting"]["utility"]["pre_completes"]:
        weight_col = "precompletion_weight"

    elif essential_columns["weighting"]["utility"]["post_completes"]:
        weight_col = "weight"
    else:
        weight_col = None
        return weight_col

    data = pd.read_parquet(data)
    if weight_col.lower() not in data.columns.tolist():
        print("Weight column not in dataset - weighting might have not been performed")
        weight_col = None

    if "weight" in data.columns.tolist() and essential_columns["weighting"]["utility"]["post_completes"]:
        weight_col = "weight"

    return weight_col


def find_conf_interval(essential_columns):
    try:
        conf_interval = essential_columns["confidence_interval"]
        return float(conf_interval)
    except:
        print("we didn't find any confidence interval in weighting")
        conf_interval = 0.95
        return conf_interval


def remove_na_strings(col):
    if col.dtypes == 'string':
        col = col.fillna('not selected')
    return col


def remove_na_strings_and_floats(col):
    if is_string_dtype(col):
        col = col.fillna('not selected')
    elif is_numeric_dtype(col):
        col = col.fillna(-999)
    return col


def remove_time_cols(df):
    not_time_cols = [x for x in df.columns if '_time' not in x]
    df = df[not_time_cols]
    df = df.apply(remove_na_strings_and_floats)
    df.columns = [x.lower() for x in df.columns]

    return df


def append_manual_seg_columns(df, manual_seg_col_df):
    manual_seg_col_df.columns = [x.lower() for x in manual_seg_col_df.columns]
    cols_to_retain = ['vrid', 'id']
    for col in manual_seg_col_df:
        if 'cint' in col:
            cols_to_retain.append(col)
    manual_seg_col_df.columns = [f'qudo_{x}' if x not in cols_to_retain else x for x in
                                      manual_seg_col_df.columns]
    df = pd.merge(df, manual_seg_col_df, left_on='cint_id', right_on='id', how='left')

    return df


def add_tgt_tag(column_list):
    new_cols = [col + '_tgt' if ('_fb' in col or '_gg' in col) and '_tgt' not in col else col for col in column_list]

    # df.columns = [col + '_tgt' if '_fb' in col or '_gg' in col and '_tgt' not in col else col for col in
    #               df.columns]

    return new_cols


def remove_numeric_tag(column_list):
    new_cols = [col.replace('_numeric', '') if '_numeric' in col else col for col in column_list]

    return new_cols

