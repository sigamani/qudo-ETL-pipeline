import logging
import random
import re
import statistics
import time
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, \
    recall_score, f1_score, precision_score, accuracy_score, cohen_kappa_score
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.contingency_tables import cochrans_q
import itertools

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def get_cluster_proportions(labels):
    """
    Function to return the proportion of the overall dataset of each cluster as a dict
    :param labels: List or np.Array. Cluster labels as integers
    :return: Dict of cluster label and proportion of overall data
    """
    proportions_dict = {}
    labels = np.array(labels)
    clusters = np.unique(labels)
    for cluster in clusters:
        try:
            proportions_dict[cluster] = np.count_nonzero(labels == cluster) / len(labels)
        except:
            print('soinbsofbasn')
    return proportions_dict


def get_cluster_metrics(data, cluster_labels, n_clusters, n_seed=np.NaN, cols_cat=None, metric=None, full_data=None,
                        calculate_overall_metrics=False):
    """
    This function takes in a dataframe, the cluster labels, metadata from the clustering method and optionally
    the category columns and whether the metric has been precomputed. It outputs a range of clustering metrics
    that can be used to assess the quality of the clusters created.
    :param data: A pandas dataframe or np.array of data that has been clustered
    :param cluster_labels: A list or 1d np.array of cluster labels
    :param n_clusters: Number of clusters
    :param n_seed: Any seed that was used in clustering
    :param cols_cat: A list of indexes of categorical columns
    :param metric: Whether the distance metric has been precomputed
    :return: a dictionary of clustering metrics along with cluster metadata
    """
    data = pd.DataFrame(data)
    dfn = data.select_dtypes(include=np.number)
    cols_num = dfn.columns.to_list()
    df_copy = data.copy()
    if not cols_cat and metric != 'precomputed':
        if not cols_cat:
            cols_cat = list(set(data.columns.to_list()) - (set(dfn.columns.to_list())))
        # minmax_scaler = MinMaxScaler()
        try:
            df_copy[cols_cat] = df_copy[cols_cat].apply(LabelEncoder().fit_transform)
        except ValueError:
            pass

    # try:
    #     df_copy[cols_num] = minmax_scaler.fit_transform(df_copy[cols_num])
    # except ValueError:
    #     pass
    df_matrix = df_copy.to_numpy()
    if not metric:
        try:
            silhouette = silhouette_score(df_matrix, cluster_labels)
        except ValueError:
            silhouette = 0
    else:
        try:
            silhouette = silhouette_score(df_matrix, cluster_labels, metric=metric)
        except ValueError:
            new_gower = df_matrix.copy()
            np.fill_diagonal(new_gower, 0)
            silhouette = silhouette_score(new_gower, cluster_labels, metric=metric)
    try:
        davies_bouldin = davies_bouldin_score(df_matrix, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(df_matrix, cluster_labels)
    except ValueError:
        davies_bouldin = 999
        calinski_harabasz = 0
    try:
        int(n_seed)
    except ValueError:
        n_seed = 'No seed selected'

    proportions = get_cluster_proportions(cluster_labels)

    if calculate_overall_metrics:
        full_data_matrix = full_data.to_numpy()
        silhouette_full = silhouette_score(full_data_matrix, cluster_labels)
        davies_bouldin_full = davies_bouldin_score(full_data_matrix, cluster_labels)
        calinski_harabasz_full = calinski_harabasz_score(full_data_matrix, cluster_labels)

        metrics = {
            'n_seed': n_seed,
            'n_clusters': int(n_clusters),
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski_harabasz,
            'silhouette_full': silhouette_full,
            'davies_bouldin_full': davies_bouldin_full,
            'calinski_harabasz_full': calinski_harabasz_full,
            'cluster_proportions': proportions
        }
    else:
        metrics = {
            'n_seed': n_seed,
            'n_clusters': int(n_clusters),
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski_harabasz,
            'cluster_proportions': proportions
        }
    return metrics


def rank_cluster_metrics(metrics_df, information_criterions=False):
    """
    Function to rank outcomes from multiple clusterings to find most performant algorithm
    :param metrics_df: Dataframe of multiple runs of get_cluster_metrics function
    :return: Dataframe of multiple runs of get_cluster_metrics function with ranks appended
    """
    metrics_df['silhouette_rank'] = metrics_df['silhouette'].rank()
    metrics_df['davies_bouldin_rank'] = metrics_df['davies_bouldin'].rank(ascending=False)
    metrics_df['calinski_harabasz_rank'] = metrics_df['calinski_harabasz'].rank()
    if information_criterions:
        metrics_df['bic_rank'] = metrics_df['bic'].rank(ascending=False)
        metrics_df['aic_rank'] = metrics_df['aic'].rank(ascending=False)
        bic_weight = 1.3
        metrics_df['bic_rank'] = metrics_df['bic_rank'] * bic_weight
        rank_cols = ['silhouette_rank', 'davies_bouldin_rank', 'calinski_harabasz_rank', 'bic_rank', 'aic_rank']
        metrics_df['rank_sum'] = metrics_df[rank_cols].sum(axis=1)
    else:
        rank_cols = ['silhouette_rank', 'davies_bouldin_rank', 'calinski_harabasz_rank']
        metrics_df['rank_sum'] = metrics_df[rank_cols].sum(axis=1)
    return metrics_df


def random_checker(df, cluster_labels, cluster_metrics, metric=None):
    """
    A method to assess whether the clustering solution is better than a random allocation of cluster labels. Returns a
    dictionary with the ratios of clustering metric to random allocations metric scores.
    :param df: Dataframe of variables used in clustering
    :param cluster_labels: A list or np.array or series of cluster labels from the clustering algorithm
    :param cluster_metrics: A dict of cluster metrics from get_cluster_metrics function
    :return: A dict of cluster metrics with clustering vs random ratios appended.
    """
    try:
        n_clusters = len(np.unique(cluster_labels))
        randoms_df = pd.DataFrame()
        for i in range(10):
            random_assignment = [random.randint(0, n_clusters) for _ in range(len(df))]
            randoms_df = pd.concat(
                [randoms_df, pd.DataFrame(get_cluster_metrics(df, random_assignment, n_clusters, metric=metric),
                                          index=[0])], axis=0, sort=False, ignore_index=True)
        random_dict = randoms_df.mean(axis=0)
        cluster_metrics['calinski_harabasz_random_ratio'] = cluster_metrics['calinski_harabasz'] / random_dict[
            'calinski_harabasz']
        cluster_metrics['davies_bouldin_random_ratio'] = random_dict['davies_bouldin'] / cluster_metrics[
            'davies_bouldin']
        cluster_metrics['silhouette_random_ratio'] = abs(cluster_metrics['silhouette'] / random_dict['silhouette'])
        return cluster_metrics
    except :
        return np.nan


def get_question_group_from_chi2_data(filtered_df, unfiltered_df):
    """
    Function to calculate the proportion of significantly different variables for each segment for a given dataset.
    Expected inputs are the outputs from ChiSquaredTester (with and without filtering), or just the _tgt variables, and
    a nested dict object is returned with a dictionary of categories for each segment.
    Parameters
    ----------
    filtered_df: Dataframe of only the significantly different variables in the format of CHiSquaredTester output
    unfiltered_df: Dataframe of all variables in the format of CHiSquaredTester output

    Returns: Nested dictionary object of proportion of significantly different variables for each category for each
        segment.
    -------

    """
    unfiltered_df['q_prefix'] = (np.where(unfiltered_df['q_code'].str.contains('_'),
                                          unfiltered_df['q_code'].str.split('_').str[0],
                                          unfiltered_df['q_code']))
    filtered_df['question_prefix'] = (np.where(filtered_df['q_code'].str.contains('_'),
                                               filtered_df['q_code'].str.split('_').str[0],
                                               filtered_df['q_code']))
    overall_counts = unfiltered_df['q_prefix'].value_counts()
    grouped = filtered_df.groupby('targeting_seg')
    seg_question_group = {}
    for segment in grouped:
        merged = pd.concat([overall_counts, segment[1]['question_prefix'].value_counts()], axis=1)
        merged['proportions'] = merged['question_prefix'] / merged['q_prefix']
        seg_question_group[segment[0]] = merged['proportions'].to_dict()
    return seg_question_group


def get_significant_variables_and_spread(chi2_data, alpha=0.05):
    """
    A function to return 4 metrics to do with the number of significantly different variables within segments and the
    categorical spread of these variables as a proportion of available variables for all questions and _tgt questions.
    Parameters
    ----------
    chi2_data: Output from ChiSquaredTester.return_api_data()
    alpha: Float. Alpha for testing significance

    Returns: Tuple. Dict of number of significantly different variables per segment, dict of number of significantly
        different _tgt variables, dict of dicts of proportion of significantly different variables within categories of
        questions for each segment, dict of dicts of proportion of significantly different _tgt variables within
        categories of _tgt questions for each segment.
    -------

    """
    sig_vars = chi2_data[0][chi2_data[0]['chi_2_result'] <= alpha]
    num_sig_vars = sig_vars['targeting_seg'].value_counts().to_dict()
    sig_tgt_vars = sig_vars[sig_vars['q_code'].str.contains("_tgt")]
    num_sig_tgt_vars = sig_tgt_vars['targeting_seg'].value_counts().to_dict()
    sig_question_groups = get_question_group_from_chi2_data(sig_vars, chi2_data[0])
    tgt_vars_only = chi2_data[0][chi2_data[0]['q_code'].str.contains("_tgt")]
    sig_tgt_question_groups = get_question_group_from_chi2_data(sig_tgt_vars, tgt_vars_only)
    return num_sig_vars, num_sig_tgt_vars, sig_question_groups, sig_tgt_question_groups


def model_consistency_checker(X_train, X_test, y_test, model, cols_cat=None):
    """
    This function aims to test the consistency of the model by splitting the data into train and test data sets, then
    refitting the model on the train dataset and predicting the labels of the test data. These are then compared to
    the original labels. Scores closer to 1 are better.
    :param X_train: Split dataset with X proportion of the dataset to retrain model
    :param X_test: Split dataset to 1-X proportion of the dataset to test retrained model
    :param y_test: outcomes from original model to test consistency against
    :param model: Model used (not fitted data)
    :return: An adjusted rand score between predicted and original labels
    """

    try:
        if isinstance(model, KPrototypes):
            cols_cat_index = [X_train.columns.get_loc(c) for c in cols_cat]
            trained_model = model.fit(X_train, categorical=cols_cat_index)
            preds = trained_model.predict(X_test, categorical=cols_cat_index)
        elif isinstance(model, BayesianGaussianMixture):
            trained_model = model.fit(X_train)
            probs = trained_model.predict_proba(X_test)
            probs_df = pd.DataFrame(probs, columns=[i for i in range(trained_model.n_components)])
            preds = probs_df.idxmax(1)
        else:
            if isinstance(X_train, list):
                print('what tha rassss')
            try:
                trained_model = model.fit(X_train)
                preds = trained_model.predict(X_test)
            except:
                return np.nan
            # if not hasattr(preds, '_labels'):
            #     probs = trained_model.predict_proba(X_test)
            #     probs_df = pd.DataFrame(probs, columns=[i for i in range(trained_model.n_components)])
            #     preds = probs_df.idxmax(1)
        return adjusted_rand_score(preds, y_test)
    except AttributeError:
        return np.nan


def label_consistency_checker(cluster_data, cluster_labels, fitted_model, cols_cat=None, metric=None):
    """
    This function tests the internal consistency of the model by retraining the model on the whole dataset and comparing
     the outputs to the original labels. Score closer to 1 are better.
    :param X_test: Split dataset to 1-X proportion of the dataset to test retrained model
    :param y_test: outcomes from original model to test consistency against
    :param fitted_model: Model that has already been fit on the whole dataset
    :return: An adjusted rand score between predicted and original labels
    """
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        sss.get_n_splits(cluster_data, cluster_labels)
        for train_idx, test_idx in sss.split(cluster_data, cluster_labels):
            train_index = train_idx

        if metric == 'precomputed':
            df = pd.DataFrame(cluster_data).iloc[train_index, train_index]
        else:
            df = pd.DataFrame(cluster_data).iloc[train_index]
        reduced_trained_labels = [cluster_labels[i] for i in train_index]

        if isinstance(fitted_model, KPrototypes):
            cols_cat_index = [df.columns.get_loc(c) for c in cols_cat]
            retrained_model = fitted_model.fit(df, categorical=cols_cat_index)

        else:
            retrained_model = fitted_model.fit(df)
        try:
            return adjusted_rand_score(retrained_model.labels_, reduced_trained_labels)
        except AttributeError:
            probs = retrained_model.predict_proba(df)
            probs_df = pd.DataFrame(probs, columns=[i for i in range(retrained_model.n_components)])
            labels = probs_df.idxmax(1)
            return adjusted_rand_score(labels, reduced_trained_labels)
    except:
        return np.nan


def get_uniqueness(chi2_data):
    """
    Function to get the uniqueness for each cluster measured by how many significantly different features appear
    only in that cluster with the same category.
    :param: chi2_data: Output from ChiSquareTester
    :return: Dict. Score between 0 and 1 for each cluster where 1 is each cluster is completely unique and 0 is all
        significantly different features appear in all clusters with the same category.
    """
    sig_df = chi2_data[0]
    cluster_sig_dfs = sig_df.groupby('targeting_seg')
    duplicate_col_names = ['q_code', 'sig_more_category']
    cluster_uniqueness = {}
    for cluster_data in cluster_sig_dfs:
        cluster = cluster_data[0]
        data = cluster_data[1]
        duplicate_df = pd.DataFrame()
        for other_clusters in cluster_sig_dfs:
            if other_clusters[0] == cluster:
                continue
            other_cluster_data = other_clusters[1]
            data['sig_more_category'] = data['sig_more_category'].apply(tuple)
            other_cluster_data['sig_more_category'] = other_cluster_data['sig_more_category'].apply(tuple)
            non_unique_df = pd.merge(data, other_cluster_data, on=duplicate_col_names, how='inner')
            duplicate_df = pd.concat([duplicate_df, non_unique_df[duplicate_col_names]], axis=0)
        duplicate_df.drop_duplicates(inplace=True)
        cluster_uniqueness[cluster] = (1 - (duplicate_df.shape[0] / data.shape[0]))
    return cluster_uniqueness


def get_communicability(chi2_data):
    """
    Function to get the communicability of each cluster measured by the count of significant creative and psychometric
    categories present.
    :param chi2_data: Output from ChiSquareTester
    :return: Dict. Keys are the clusters, Values are the count of creative/psychometric metrics alongside the average.
    """
    sig_df = chi2_data[0]
    comm_categories = 'psy|ae'
    try:
        comm_df = sig_df[sig_df['q_code'].str.contains(comm_categories)].copy()
    except AttributeError:
        print(chi2_data)
    comm_grouped = comm_df.groupby(['targeting_seg'])['q_code'].nunique().reset_index()
    comm_grouped.rename(columns={'q_code': 'communicability'}, inplace=True)
    comm_dict = dict()
    comm_dict['data'] = dict(zip(comm_grouped['targeting_seg'], comm_grouped['communicability']))
    try:
        comm_dict['avg'] = comm_grouped['communicability'].sum() / sig_df['targeting_seg'].nunique()
    except statistics.StatisticsError:
        comm_dict['avg'] = np.nan
    return comm_dict


def get_social_presence(data, cluster_labels, sm_platform):
    """
    This function returns the proportion of people in a cluster using FB (for now)
    on a regular basis.
    :param data: dataframe of survey results
    :param cluster_labels: cluster labels in int format
    :param sm_platform: FB, Google
    :return: Social media presence per cluster (as a decimal)
    """

    """ Identify relevant question column to use."""
    _col = ''

    for column in data:
        if "mc_" in column:
            column_values = data[column].values
            try:
                for v in column_values:
                    if sm_platform.lower() in str(v).lower():
                        _col = column
                        break
            except TypeError:
                pass
    if _col == '':
        return np.nan
    print(f"Calculating social media presence using {sm_platform} and the {_col} column")

    """ Calculate percentage of social media use per cluster. TODO: Add weighting. """
    data['cluster'] = cluster_labels
    labels = list(set(cluster_labels))
    results = {}

    for _l in labels:
        if _col == '':
            break
        _data = data[data['cluster'] == _l]
        cluster_size = len(_data)
        count = len(_data[_data[_col] == sm_platform])
        results[_l] = round(count / cluster_size, 3)

    return results


def data_slicer(data, tgt_cols, pop_modes):
    """
    Function to extract signal loss for every cluster or the segment based on the target_columns in the input.
    :param data: it is the original data before encoding it (strings or floats)
    :param tgt_cols: columns named for tgt
    :param pop_modes: the responses for every tgt_column
    :return: data_frame filtered based on the tgt_cols and Pop modes, the core_cols tgt_cols, the sub_cols ones, and the
    target of the return to return as minimum 25% of input data after slicing
    """
    percent_retained = 0.25
    min_count = percent_retained * data.shape[0]
    core_cols = []
    while len(core_cols) < 6:
        if percent_retained < 0.15:
            break
        core_cols = []
        sub_cols = []
        if percent_retained < 0.15:
            break
        for col, mode in zip(tgt_cols, pop_modes):
            data_prev = data.copy()
            data = data[data[col] == mode]
            if data.shape[0] < min_count:
                data = data_prev.copy()
                sub_cols.append(col)
            else:
                core_cols.append(col)
                continue
        percent_retained -= 0.01
        min_count = percent_retained * data.shape[0]
    return data, core_cols, sub_cols, percent_retained


def calc_chi_square_signal(data, original_data, seg):
    if data.shape[0] == 0:
        return {'signal': 0, 'precision': 0, 'recall_score': 0, 'f1_score': 0}
    vc = data['labels'].value_counts()
    vc = vc.reset_index()
    vc.columns = ['labels', 'counts']
    true = original_data[original_data['labels'] == seg].shape[0]
    false = original_data[original_data['labels'] != seg].shape[0]
    try:
        tp = float(vc[vc['labels'] == seg]['counts'])
        fp = np.sum(vc['counts']) - tp
        fn = true - tp
        tn = false - (np.sum(vc['counts']) - tp)
        accuracy = round((tp + tn) / (tp + tn + fp + fn), 3)
        precision = round((tp / (tp + fp)), 3)
        recall = round((tp / (tp + fn)), 3)
        f1_score = round((2 * (precision * recall) / (precision + recall)), 3)
        return {'signal': accuracy, 'precision': precision, 'recall_score': recall,
                'f1_score': f1_score}
    except:
        return {'signal': 0, 'precision': 0, 'recall_score': 0, 'f1_score': 0}


def get_signal_loss_chi_square(original_data, cluster_labels, chi2_data):
    """
    Function to extract signal loss for every cluster or the segment based on the target_columns in the input.
    :param orginal_data: it is the original data before encoding it (strings or floats)
    :param cluster_labels: A list or 1d np.Array of cluster labels
    :param chi2_data: Tuple. Two items, deliver data (dataframe) and discover data (list of dicts). See ChiSquareTester
        for more details.
    :return: a dictionary of cluster or target segment, recall score, signal loss (accuracy),
    precision score and F1_score
    """
    if len(chi2_data) == 0:
        return np.nan
    if len(cluster_labels) == 0:
        return np.nan
    if original_data.shape[0] == 0:
        return np.nan
    if isinstance(cluster_labels, pd.Series):
        original_data['labels'] = cluster_labels.to_list()
    else:
        original_data['labels'] = cluster_labels
    results = dict()
    cols_cols = dict()
    core_results = dict()
    percent_retained = dict()
    chi2_data = chi2_data[0]
    unique_segments = np.unique(original_data['labels'])
    unique_segments = list(unique_segments)
    for seg in unique_segments:
        try:
            chi2 = chi2_data[chi2_data['unencoded_seg'] == str(seg)][['q_code', 'pop_mode']]
        except:
            chi2 = chi2_data[chi2_data['targeting_seg'] == str(seg)][['q_code', 'pop_mode']]

        tgt_cols = [col for col in chi2['q_code'] if 'tgt' in col.lower()]
        if len(tgt_cols) == 0:
            gg_list = [col for col in chi2['q_code'] if '_gg' in col.lower()]
            fb_list = [col for col in chi2['q_code'] if '_fb' in col.lower()]
            in_gg_but_not_in_fb = set(fb_list) - set(gg_list)
            tgt_cols = gg_list + list(in_gg_but_not_in_fb)

        pop_modes = chi2[chi2['q_code'].isin(tgt_cols)]['pop_mode']
        data = original_data[tgt_cols].copy()
        data['labels'] = cluster_labels
        seg_data = data[data['labels'] == seg]
        sliced_data, core_cols, sub_cols, percent_retained_seg = data_slicer(seg_data, tgt_cols=tgt_cols,
                                                                             pop_modes=pop_modes)
        percent_retained[str(seg)] = percent_retained_seg
        cols_cols[str(seg)] = core_cols
        results[str(seg)] = calc_chi_square_signal(data, original_data, seg)
        core_results[str(seg)] = calc_chi_square_signal(sliced_data, original_data, seg)

    return results, cols_cols, core_results, percent_retained


def n_cols_cleaner(cols_list, noise_list):
    cols = cols_list
    for noise in noise_list:
        cols = col_cleaner(cols, noise)
    cleaned_list = cols
    return cleaned_list


def col_cleaner(cols_list, noise):
    cleaned_cols = []
    for col in cols_list:
        cleaned = col.replace(noise, "")
        cleaned_cols.append(cleaned)
    return cleaned_cols


def commen_word_exrtactor(s1, s2):
    list_s1 = s1.split("_")
    list_s2 = s2.split("_")
    common = [word for word in list_s1 if word in list_s2]
    lengths = [len(word) for word in common]
    if len(common) == 0:
        return common, False
    else:
        return common[lengths.index(max(lengths))], True


def complex_regex_extractor(tgt_cols):
    noise_list = ['tgt', 'sbeh', 'life', 'mc', 'uk', 'cb', 'ww', 'gg', 'ims', 'fb']
    tgt_cols = n_cols_cleaner(tgt_cols, noise_list)
    commons = []
    for counter in range(len(tgt_cols) - 1):
        if counter + 1 >= len(tgt_cols):
            common_result, cond = commen_word_exrtactor(tgt_cols[counter], tgt_cols[counter + 1])
            if cond is False:
                break
            else:
                if len(common_result) <= 3:
                    continue
                commons.append(common_result)
        else:
            common_result, cond = commen_word_exrtactor(tgt_cols[counter], tgt_cols[counter + 1])
            if cond is False:
                break
            else:
                if len(common_result) <= 3:
                    continue
                commons.append(common_result)
    commons = list(np.unique(commons))
    return commons


def simple_regex_extractor(tgt_cols):
    keeper = []
    for col in tgt_cols:
        _l = col.split('_')
        keeper.append(_l[2])
    cat_cols = list(set(keeper))
    return cat_cols


def get_variability(encoded_data, cluster_labels):
    """
    Function to calculate the magnitude for all segments
    :param encoded_data: DataFrame for all original data(self.encoded_data+cluster_labels) with the labels
    :param cluster_labels: A list or 1d np.Array of cluster labels
    :return: a float number for variability or heterogeneity
    """
    if encoded_data.shape[0] == 0:
        return np.nan
    if len(cluster_labels) == 0:
        return np.nan
    encoded_data['labels'] = cluster_labels
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    unique_clusters = list(unique_clusters)
    unique_clusters.sort()
    columns_tgt = [col for col in encoded_data.columns if "tgt" in col.lower()]
    if len(columns_tgt) == 0:
        return np.nan
    else:
        results = {}

        try:
            q_codes = list(dict.fromkeys([re.findall(r"(\d{4,})", col)[0] for col in columns_tgt]))
        except IndexError:
            q_codes = complex_regex_extractor(tgt_cols=columns_tgt)
        tgt_categories = []
        i2_scores = []
        for q_code in q_codes:
            tgt_categories.append([col for col in encoded_data.columns if "_" + q_code in col])
        for category in tgt_categories:
            category_data = encoded_data[category]
            try:
                category_cochran = cochrans_q(category_data).statistic
            except:
                continue
            df = cochrans_q(category_data).df
            i2 = round(((category_cochran - df) / category_cochran) * 100, 3)
            if i2 > 0:
                i2_scores.append(i2)
        results['all_clusters'] = round(np.mean(i2_scores), 3)

        for cluster in unique_clusters:
            i2_score_cluster = []
            cluster_data = encoded_data[encoded_data['labels'] == cluster]
            for category in tgt_categories:
                category_data_cluster = cluster_data[category]
                try:
                    category_cochran_cluster = cochrans_q(category_data_cluster).statistic
                except:
                    continue
                df_cluster = cochrans_q(category_data_cluster).df
                i2_cluster = round(((category_cochran_cluster - df_cluster) / category_cochran_cluster) * 100, 3)
                if i2_cluster > 0:
                    i2_score_cluster.append(i2_cluster)
            results[cluster] = round(np.mean(i2_score_cluster), 3)
    return results


def get_magnitude(chi2_data, cluster_labels, encoded_data):
    """
    Function to calculate the magnitude for all segments
    :param chi2_data: Tuple. Two items, deliver data (dataframe) and discover data (list of dicts). See ChiSquareTester
        for more details.
    :param cluster_labels: A list or 1d np.Array of cluster labels
    :param encoded_data: Original data to be compared to (not currently implemented)
    :return: a float number for magnitude

    """

    if len(chi2_data) == 0:
        return np.nan
    if len(cluster_labels) == 0:
        return np.nan
    if encoded_data.shape[0] == 0:
        return np.nan
    columns_tgt = [col for col in encoded_data.columns if "tgt" in col.lower()]
    num_tgt = len(columns_tgt)
    if num_tgt == 0:
        return np.nan
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    if isinstance(cluster_labels, list):
        cluster_labels = np.array(cluster_labels)
    percentages = [round((count / cluster_labels.shape[0]), 2) for count in list(counts)]
    unique_clusters = list(unique_clusters)
    counting_dict = {'percentages': percentages,
                     'cluster_seg': unique_clusters}
    magnitude = pd.DataFrame(counting_dict)
    magnitude = magnitude.sort_values(by='cluster_seg')
    chi2_df = chi2_data[0]
    chi2_tgt = [col for col in chi2_df['q_code'] if 'tgt' in col.lower()]
    chi2_df = chi2_df[chi2_df['q_code'].isin(chi2_tgt)]
    tgt_sig = chi2_df['targeting_seg'].value_counts()
    tgt_sig = tgt_sig.reset_index()
    tgt_sig.columns = ['cluster_sig', 'counts']
    tgt_sig['ratio_sig'] = tgt_sig['counts'] / num_tgt
    tgt_sig = tgt_sig.sort_values(by='cluster_sig')
    tgt_sig.reset_index(inplace=True)
    magnitude = pd.concat([magnitude, tgt_sig], axis=1)
    magnitude['magnitude'] = magnitude['ratio_sig'] * magnitude['percentages']
    magnitude['magnitude'] = magnitude['magnitude'].round(decimals=3)

    if magnitude['magnitude'].sum() > 0:
        if magnitude['magnitude'].sum() <= 1:
            output = dict(zip(magnitude['cluster_seg'], magnitude['magnitude']))
            output['all_clusters'] = round(magnitude['magnitude'].sum(), 3)
            return output
        else:
            return np.nan
    else:
        return np.nan


def get_message_reach_metric(social_presence, signal_loss):
    """
    Function to extract message reach based on signal and presence for every segment.
    :param social_presence: dict with int keys and values from 0 to 1 for social presence.
    :param signal_loss: dict with str keys and values of dict formed from signal, f1_score, precision and recall.
    :return: dict of str keys of labels and the multiplication of signal and presence, the value should be
    between 0 and 1.
    """
    if len(signal_loss) > len(social_presence):
        signal_loss.pop('all')
    if len(signal_loss) != len(social_presence):
        return np.nan
    if len(signal_loss) == 0 or len(social_presence) == 0:
        return np.nan
    results = dict()
    labels = np.unique(list(signal_loss.keys()))
    labels = [int(label) for label in labels]
    labels.sort()
    for label in labels:
        signal = signal_loss[str(label)]
        results[str(label)] = round((social_presence[label] * signal['signal']), 3)
    return results


def get_signal_loss_metric(data_encoded, cluster_labels, target_column='clusters', sampling=None):
    """
    Function to extract signal loss for every cluster or the segment based on the target_columns in the input.
    :param data_encoded: it is the original data after encoding it
    :param cluster_labels: A list or 1d np.Array of cluster labels
    :param target_column: A string for the column name of the target
    :param sampling: A string to specify which sampling method to be used
    :return: a dictionary of cluster or target segment, recall score, signal loss (accuracy),
    precision score, F1_score and Cohen Kappa Score
    """
    if data_encoded.shape[0] == 0:
        return np.nan
    if len(cluster_labels) == 0:
        return np.nan
    data_encoded.reset_index(inplace=True, drop=True)
    data_encoded['clusters'] = cluster_labels
    if target_column not in data_encoded.columns:
        return np.nan

    result = dict()
    unique_segments, counts = np.unique(data_encoded[target_column], return_counts=True)
    tgt = [col for col in data_encoded.columns if 'tgt' in col.lower()]
    if len(tgt) == 0:
        return np.nan

    features = data_encoded[tgt]
    target = data_encoded[target_column]

    if sampling == 'under':
        rus = RandomUnderSampler(random_state=42, replacement=True)
        features_final, target_final = rus.fit_resample(features, target)
    elif sampling == 'over':
        ros = RandomOverSampler(random_state=42)
        features_final, target_final = ros.fit_resample(features, target)
    elif sampling == 'smote':
        smote = SMOTE(random_state=42)
        features_final, target_final = smote.fit_resample(features, target)
    else:
        features_final = features
        target_final = target

    X_train, X_test, y_train, y_test = train_test_split(features_final, target_final, test_size=0.33, random_state=42)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    signal_loss = round(accuracy_score(y_test, y_pred), 3)
    f1_scores = round(f1_score(y_test, y_pred, average='weighted'), 3)
    precision = round(precision_score(y_test, y_pred, average='weighted'), 3)
    recall_scores = round(recall_score(y_test, y_pred, average='weighted'), 3)
    cohen_kappa_scores = round(cohen_kappa_score(y_test, y_pred), 3)  # need to inspect
    result['all'] = {'signal': signal_loss, 'f1_score': f1_scores, 'precision': precision,
                     'recall_score': recall_scores, 'cohen_kappa_score': cohen_kappa_scores}
    for segment in unique_segments:
        trial_data = data_encoded[data_encoded[target_column] == segment]
        x_loop = trial_data[tgt]
        y_loop = trial_data[target_column]
        y_pred_loop = rf.predict(x_loop)
        signal_loss = round(accuracy_score(y_loop, y_pred_loop), 3)
        f1_scores = round(f1_score(y_loop, y_pred_loop, average='weighted'), 3)
        precision = round(precision_score(y_loop, y_pred_loop, average='weighted'), 3)
        recall_scores = round(recall_score(y_loop, y_pred_loop, average='weighted'), 3)
        cohen_kappa_scores = round(cohen_kappa_score(y_loop, y_pred_loop), 3)
        result[str(segment)] = {'signal': signal_loss, 'f1_score': f1_scores, 'precision': precision,
                                'recall_score': recall_scores, 'cohen_kappa_score': cohen_kappa_scores}
    return result


def get_best_signal_metric(data_encoded, cluster_labels, target_column='clusters'):
    """
    Function to evaluate overall signal metrics from model trained using over and under sampling and output the best method.
    :param data_encoded: it is the original data after encoding it
    :param cluster_labels: A list or 1d np.Array of cluster labels
    :param target_column: A string for the column name of the target
    :return: the best dictionary of cluster or target segment, recall score, signal loss (accuracy),
    precision score, F1_score and Cohen Kappa Score
    """

    comparison_dict = dict()

    comparison_dict['rus'] = get_signal_loss_metric(data_encoded=data_encoded.copy(), cluster_labels=cluster_labels,
                                                    target_column=target_column, sampling='under')
    comparison_dict['ros'] = get_signal_loss_metric(data_encoded=data_encoded.copy(), cluster_labels=cluster_labels,
                                                    target_column=target_column, sampling='over')
    comparison_dict['none'] = get_signal_loss_metric(data_encoded=data_encoded.copy(), cluster_labels=cluster_labels,
                                                     target_column=target_column)

    comparison_list = []
    for k, v in comparison_dict.items():
        temp_df = pd.DataFrame.from_dict(v['all'], orient='index')
        temp_df.columns = [k]
        comparison_list.append(temp_df)

    comparison_df = pd.concat(comparison_list, axis=1)
    comparison_df['winner'] = comparison_df.idxmax(axis=1)
    winner = comparison_df['winner'].value_counts().index.tolist()[0]
    logging.info(f'Best signal metrics produced via samping method: {winner}')
    return comparison_dict[winner]


def get_all_metrics(cluster_data, cluster_labels, n_clusters, model, fitted_model, algo,
                    n_seed=np.NaN, cols_cat=None, metric=None, chi2_data=None, full_data=None,
                    data_encoded=None):
    """
    Function to provide all metrics associated with clustering algorithms and pass these back to the calling function
    :param cluster_data: A pandas dataframe or np.Array of data that has been clustered
    :param cluster_labels: A list or 1d np.Array of cluster labels
    :param n_clusters: Number of clusters
    :param n_seed: Any seed that was used in clustering
    :param cols_cat: A list of indexes of categorical columns (default None)
    :param metric: Whether the distance metric has been precomputed (default None)
    :param chi2_data: Tuple. Two items, deliver data (dataframe) and discover data (list of dicts). See ChiSquareTester
        for more details.
    :param model: SKLearn-type model that was used
    :param fitted_model: Fitted version of model
    :param algo: Str. Name of algorithm
    :param full_data: Original data without encoding or filtering
    :param data_encoded: it is the original data after encoding it
    :return: a dictionary of clustering metrics along with cluster metadata
    """

    metrics = get_cluster_metrics(cluster_data, cluster_labels, n_clusters, n_seed=n_seed, cols_cat=cols_cat,
                                  metric=metric, full_data=data_encoded, calculate_overall_metrics=True)
    metrics = random_checker(cluster_data, cluster_labels, metrics, metric=metric)
    fb_social_presence = get_social_presence(full_data, cluster_labels, sm_platform='Facebook')
    metrics['fb_presence'] = fb_social_presence
    """Todo: add gg presence"""
    # metrics['gg_presence'] = get_social_presence(full_data, cluster_labels, sm_platform='Google')

    consistency_start = time.time()
    if hasattr(model, 'predict') and not isinstance(model, BayesianGaussianMixture):
        model_for_testing = clone(model)
        X_train, X_test, y_train, y_test = train_test_split(cluster_data, cluster_labels, test_size=0.1,
                                                            random_state=42,
                                                            stratify=cluster_labels)
        if cols_cat:
            metrics['model_consistency'] = model_consistency_checker(X_train, X_test, y_test, model_for_testing, cols_cat)
        else:
            metrics['model_consistency'] = model_consistency_checker(X_train, X_test, y_test, model_for_testing)
    else:
        metrics['model_consistency'] = np.nan
    consistency_end = time.time()
    print(f'Model consistency took {consistency_end - consistency_start} seconds to execute')
    labelling_start = time.time()
    if fitted_model is None or isinstance(model, BayesianGaussianMixture):
        metrics['label_consistency'] = np.nan
    else:
        try:
            metrics['label_consistency'] = label_consistency_checker(cluster_data, cluster_labels, fitted_model,
                                                                      cols_cat=cols_cat, metric=metric)
        except ValueError:
            logging.warning('Unable to test label consistency due to small clusters')
            metrics['label_consistency'] = np.nan
    labelling_end = time.time()
    print(f'Model relabel took {labelling_end - labelling_start} seconds to execute')

    if chi2_data and not chi2_data[0].empty:
        metrics['uniqueness'] = get_uniqueness(chi2_data)
        communicability_metrics = get_communicability(chi2_data)
        metrics['communicability_clusters'] = communicability_metrics['data']
        metrics['communicability_average'] = communicability_metrics['avg']
        significant_variables_metrics = get_significant_variables_and_spread(chi2_data)
        metrics['significant_variables'] = significant_variables_metrics[0]
        metrics['significant_tgt_variables'] = significant_variables_metrics[1]
        metrics['spread_of_significant_variables'] = significant_variables_metrics[2]
        metrics['spread_of_significant_tgt_variables'] = significant_variables_metrics[3]
    else:
        metrics['uniqueness'] = np.nan
        metrics['communicability_clusters'] = np.nan
        metrics['communicability_average'] = np.nan
    metrics['magnitude'] = get_magnitude(chi2_data, cluster_labels, encoded_data=data_encoded.copy())
    metrics['variability'] = get_variability(data_encoded.copy(), cluster_labels)
    if [x for x in data_encoded.columns if 'tgt' in x]:
        ml_signal_loss = get_best_signal_metric(data_encoded=data_encoded.copy(), cluster_labels=cluster_labels,
                                                target_column='clusters')
        metrics['ml_signal'] = ml_signal_loss

        signal, core_columns, optimal_signal, percent_retained = get_signal_loss_chi_square(
            original_data=full_data.copy(),
            cluster_labels=cluster_labels,
            chi2_data=chi2_data)
        metrics['chi2_signal'] = signal
        try:
            if len(fb_social_presence) == n_clusters:
                message_reach_ml = get_message_reach_metric(social_presence=fb_social_presence, signal_loss=ml_signal_loss)
                metrics['message_reach_ml_signal'] = message_reach_ml
                metrics['massage_reach_chi2_signal'] = get_message_reach_metric(social_presence=fb_social_presence,
                                                                                signal_loss=signal)
                metrics['chi2_signal_core_columns'] = optimal_signal
                metrics['message_reach_optimal_signal'] = get_message_reach_metric(social_presence=fb_social_presence,
                                                                                   signal_loss=optimal_signal)
        except TypeError:
            pass
        metrics['core_columns'] = core_columns
        metrics['percent_retained_for_core_cols'] = percent_retained
    metrics['algorithm'] = algo

    return metrics
