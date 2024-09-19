import logging
import multiprocessing
import sys
import threading
import time
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from joblib import Parallel, delayed
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from rpy2.rinterface import embedded
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.base import clone
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tqdm import tqdm

# try:
from kraken.app.metrics import get_cluster_metrics, rank_cluster_metrics, get_all_metrics
from kraken.app.inference.chisquared_tester import ChiSquaredTester
# except ModuleNotFoundError:
#     from .metrics import get_cluster_metrics, rank_cluster_metrics, get_all_metrics
#     from .inference.chisquared_tester import ChiSquaredTester

# Threadding to be used in timeout for LCA and DBScan functions
try:
    import thread
except ImportError:
    import _thread as thread

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def cdquit(fn_name):
    """
    Function to exit the thread of a specific function
    :param fn_name: name of function
    """
    # print to stderr, unbuffered in Python 2.
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt


def exit_after(s):
    """
    Use as decorator to exit process if function takes longer than s seconds
    :param s: Int. Number of seconds
    """

    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, cdquit, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result

        return inner

    return outer


class Clusterings:
    """
    Performs all manor if clusterings using a variety of algorithms as part of the Kraken process. It takes in cleaned
    questionnaire data, and returns cluster labels and a collection of metrics to assess clustering quality.

    Parameters
    ----------
    data: pd.DataFrame
        Cleaned and processed response data

    cluster_vars: List
        A list of column names to be included in the clustering process

    Attributes
    ----------

    """

    def __init__(self,
                 survey_name,
                 data,
                 cluster_vars,
                 full_data=None,
                 hierarchical=None,
                 subsample=None,
                 ignore_hierarchical_value=None,
                 weight_col=None,
                 conf_interval=0.95):
        self.survey_name = survey_name
        self.conf_interval = conf_interval
        self.data = data.apply(pd.to_numeric, errors='ignore')
        self.cluster_vars = cluster_vars
        self.data_encoded = data.apply(LabelEncoder().fit_transform)
        if isinstance(full_data, pd.DataFrame):
            self.full_data = full_data
            self.full_data_encoded = full_data.apply(LabelEncoder().fit_transform)
        else:
            self.full_data = self.data.copy()
            self.full_data_encoded = data.apply(LabelEncoder().fit_transform)
        self.min_k = 3
        self.max_k = 9
        self.kprototypes_or_kmodes()
        self.get_categorical_cols()
        self.num_cores = multiprocessing.cpu_count()
        self.opt_clusters = None
        self.seeds = [1, 42, 100, 200, 404, 500, 1000, 123, 321, 78]
        self.upper_cluster_threshold = 0.55
        self.lower_cluster_threshold = 0.02
        self.hierarchical = hierarchical
        self.subsample = subsample
        self.ignore_hierarchical_value = ignore_hierarchical_value
        self.hierarchical_grouped_by = None
        if self.hierarchical:
            self.set_hierachical_min_max_k()
            self.set_hierachical_grouped_by()
            if self.ignore_hierarchical_value:
                self.set_ignore_hierarchical_value()

        self.weight_column = weight_col

        # todo: I do not recommend this as this does not worka s intended
        # if self.weight_col:
        #     self.weight = self.data[self.weight_col]
        # else:
        #     try:
        #         self.weight = self.data['weight']
        #     except KeyError:
        #         self.weight = None

    def set_ignore_hierarchical_value(self):
        if not isinstance(self.ignore_hierarchical_value, list):
            self.ignore_hierarchical_value = [self.ignore_hierarchical_value]
        encoded_vals_to_ignore = []
        for val in self.ignore_hierarchical_value:
            idx_of_ignored_value = self.data[self.hierarchical][self.data[self.hierarchical] ==
                                                                val].index[0]
            encoded_vals_to_ignore.append(self.data_encoded[self.hierarchical].iloc[idx_of_ignored_value])
        self.ignore_hierarchical_value = encoded_vals_to_ignore

    def set_hierachical_grouped_by(self):
        self.hierarchical_grouped_by = self.data_encoded.groupby(self.hierarchical)

    def set_hierachical_min_max_k(self):
        self.min_k = 2
        self.max_k = 3

    def get_chisquare_data(self,
                           labels,
                           weights=None,
                           correction='bonferroni',
                           segmentation=None,
                           rename_segments=None,
                           clustered_df=None):

        survey_name = self.survey_name

        if not isinstance(clustered_df, pd.DataFrame):
            clustered_df = self.full_data
        clustered_df.replace({'None': None}, inplace=True)
        if isinstance(labels, pd.Series):
            clustered_df['cluster'] = labels.to_list()
        else:
            clustered_df['cluster'] = labels
        chi2_tester = ChiSquaredTester(survey_name=survey_name,
            segmentation=segmentation,
            clustered_data=clustered_df,
            seg_col='cluster',
            rename_segments=rename_segments,
            conf_interval=self.conf_interval,
            weights=weights,
            correction=correction
        )
        data = chi2_tester.return_API_data()
        try:
            data_and_labels = pd.concat([rename_segments.reset_index(drop=True), pd.Series(labels, dtype="string")],
                                        axis=1)
            data_and_labels.columns = ['data', 'label']
            uniques = data_and_labels.groupby(['data', 'label']).size().reset_index()
            uniques.set_index('label', inplace=True)
            cluster_mapping = uniques.to_dict()
            data[0]['unencoded_seg'] = data[0]['targeting_seg'].copy()
            data[0].replace({"targeting_seg": cluster_mapping['data']}, inplace=True)
            for cluster in data[1]:
                if isinstance(cluster, pd.DataFrame):
                    cluster[cluster_mapping['data'][list(cluster.keys())[0]]] = cluster[list(cluster.keys())[0]]
                    del cluster[list(cluster.keys())[0]]
            for cluster in data[2]:
                try:
                    cluster[cluster_mapping['data'][list(cluster.keys())[0]]] = cluster[list(cluster.keys())[0]]
                except KeyError:
                    pass
                del cluster[list(cluster.keys())[0]]
        except (NameError, AttributeError) as e:
            pass
        return data

    def get_categorical_cols(self):
        """
        Method to identify which columns in the dataset contain categorical columns and sets this list as an attribute
        on self.
        :return:
        """
        if self.cluster_vars:
            df = self.data[self.cluster_vars]
        else:
            df = self.data
        dfn = df.select_dtypes(include=np.number)
        self.categorical_cols = list(set(df.columns.to_list()) - (set(dfn.columns.to_list())))

    def kprototypes_or_kmodes(self):
        """
        Method to identify whether Kmodes or Kprototypes is the appropriate algorithm to use for the data. Saves
        as a string attribute
        :return:
        """
        if self.cluster_vars:
            df = self.data[self.cluster_vars]
        else:
            df = self.data
        dfn = df.select_dtypes(include=np.number)
        if dfn.empty:
            self.method_kprototypes_or_kmodes = 'kmodes'
        else:
            self.method_kprototypes_or_kmodes = 'kprototypes'

    def standardize_data(self, data):
        """
        Method to standardise numeric data columns to use in PCA and Kmeans
        :param data: pd.DataFrame or np.Array of numeric data to standardise
        :return: pd.DataFrame of standardised data
        """
        if not self.categorical_cols:
            self.kprototypes_or_kmodes()
        scaler = StandardScaler()
        if self.categorical_cols and self.method_kprototypes_or_kmodes == 'kprototypes':
            num_cols = data.columns[~data.columns.isin(self.categorical_cols)]
            scaler.fit(data[num_cols])
            std_data = scaler.transform(data[num_cols])
            categorical = data[self.categorical_cols]
            scaler = StandardScaler()
            categorical = scaler.fit_transform(categorical)
            std_data = pd.concat([pd.DataFrame(std_data, columns=num_cols), pd.DataFrame(categorical)], axis=1)
        else:
            data = data.replace('not selected', 0)
            scaler.fit(data)
            std_data = scaler.transform(data)
        return std_data, scaler

    def find_n_components(self, data):
        """
        Method to find the optimal number of components for a given dataset for PCA analysis
        :param data: pd.DataFrame or np.Array
        :return: integer of optimal components to use in PCA
        """
        pca = PCA(n_components=data.shape[1], svd_solver='auto', random_state=42)
        pca.fit(data)
        cumsums = pca.explained_variance_ratio_.cumsum()
        for i, cumsum in enumerate(cumsums):
            if cumsum < 0.5:
                pass
            else:
                ideal_n = i + 1
                break
        return ideal_n, pca

    def get_pca_data(self, std_data, find_components=True, n_components=None):
        """
        Method to get the Principal Components Analysis transformed data from the dataset
        :param std_data: pd.DataFrame or np.Array of standardised data
        :param find_components: Bool, whether to find the optimal components (Default = True)
        :param n_components: Optional Int, number of components to use is find_components is False (Default = None)
        :return: np.Array of PCA transformed data and number of components used
        """
        if find_components:
            n, pca = self.find_n_components(data=std_data)
        else:
            n = n_components
            pca = PCA(n_components=n, svd_solver='auto', random_state=42)
        pca_data = pca.fit_transform(std_data)
        self.optimal_pca_components = n
        return pca_data, n, pca

    def kmeans_and_pca(self, data, k, n_components=None, seed=42):
        """
        Method to perform kmeans on pca transformed data.
        :param data: pd.DataFrame of response data
        :param k: Int. Number of clusters
        :param n_components: Int. Number of components used in PCA (default None)
        :param seed: Int. Seed used in Kmeans algorithm (default 42 - as it tradition)
        :return: Fitted Kmeans model, number of components, and PCA transformed data
        """
        if isinstance(data, pd.DataFrame):
            data, scaler = self.standardize_data(data)
        if n_components:
            pca_data, n_components, pca = self.get_pca_data(data, find_components=False, n_components=n_components)
        else:
            pca_data, n_components, pca = self.get_pca_data(data)
        kmeans = KMeans(n_clusters=k, random_state=seed)
        try:
            kmeans.fit_predict(pca_data)
        except ValueError:
            pass
        return kmeans, n_components, pca_data, pca

    def get_pca_and_kmeans(self, data, k, n_components=None, seed=42):
        """
        Method to call kmeans_and_pca and to gather metrics on the results
        :param data: pd.DataFrame or np.Array of questionnaire responses
        :param k: Int. Number of clusters to be used in Kmeans
        :param n_components: Int. Number of components to be used in PCA (default None)
        :param seed: Int. Seed used in Kmeans algorithm (default 42 - as it tradition)
        :return: Fitted kmeans model and dict of metrics
        """
        kmeans, n_components, pca_data, pca = self.kmeans_and_pca(data, k, n_components=n_components, seed=seed)
        metrics = get_cluster_metrics(pca_data, kmeans.labels_, k, n_seed=seed)
        return kmeans, metrics, pca_data, pca

    def kprototypes_cost(self, num_clusters, cluster_df, verbose_binary):
        """
        Method to calcuate the cost of clustering using Kprototypes (used in assessing optimal clusters)
        :param num_clusters: Int. Number of clusters to use
        :param cluster_df: pd.DataFrame or np.Array of questionniaire responses
        :param verbose_binary: Bool. Whether to verbosely print to log
        :return: float of clustering cost and dict of cluster metrics
        """
        cols_cat_index = [cluster_df.columns.get_loc(c) for c in self.categorical_cols if c in cluster_df.columns]
        try:
            kproto = KPrototypes(
                n_clusters=num_clusters, init="Cao", n_init=1, verbose=verbose_binary, n_jobs=-1, random_state=42
            )
            kproto.fit_predict(cluster_df, categorical=cols_cat_index)
            cluster_metrics = get_cluster_metrics(cluster_df, kproto.labels_, num_clusters, n_seed=42)
        except ValueError:
            return
        print(f"finalising kprototypes_cost function {num_clusters}...")
        return kproto.cost_, cluster_metrics

    @staticmethod
    def kmodes_cost(num_clusters, cluster_df, verbose_binary):
        """
        Method to calcuate the cost of clustering using Kmodes (used in assessing optimal clusters)
        :param num_clusters: Int. Number of clusters to use
        :param cluster_df: pd.DataFrame or np.Array of questionniaire responses
        :param verbose_binary: Bool. Whether to verbosely print to log
        :return: float of clustering cost and dict of cluster metrics
        """
        kmode = KModes(
            n_clusters=num_clusters, init="Cao", n_init=1, verbose=verbose_binary, n_jobs=-1
        )
        kmode.fit_predict(cluster_df)
        cluster_metrics = get_cluster_metrics(cluster_df, kmode.labels_, num_clusters)
        print(f"finalising kmodes_cost function {num_clusters}...")
        return kmode.cost_, cluster_metrics

    @staticmethod
    def kmeans_inertia(num_clusters, cluster_df, verbose_binary):
        """
        Method to calcuate the inertia of clustering using Kmeans (used in assessing optimal clusters)
        :param num_clusters: Int. Number of clusters to use
        :param cluster_df: pd.DataFrame or np.Array of questionniaire responses
        :param verbose_binary: Bool. Whether to verbosely print to log
        :return: float of clustering inertia and dict of cluster metrics
        """
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, verbose=verbose_binary).fit_predict(cluster_df)
        metrics = get_cluster_metrics(cluster_df, kmeans.labels_, num_clusters)
        return kmeans.inertia_, metrics

    def kmeans_and_pca_inertia(self, num_clusters, cluster_df, verbose_binary, n_components):
        """
        Method to calcuate the inertia of clustering using Kmeans and PCA (used in assessing optimal clusters)
        :param num_clusters: Int. Number of clusters to use
        :param cluster_df: pd.DataFrame or np.Array of questionnaire responses
        :param verbose_binary: Bool. Whether to verbosely print to log
        :param n_components: Int. Number of components to use in PCA
        :return: float of clustering cost and dict of cluster metrics
        """
        kmeans, n_components, pca_data, pca = self.kmeans_and_pca(cluster_df, num_clusters, n_components=n_components)
        metrics = get_cluster_metrics(pca_data, kmeans.labels_, num_clusters)
        return kmeans.inertia_, metrics, n_components

    def optimal_clusters(self, k_increment=1, method=None, cluster_df=None, hierarchical=False):
        """
        Method to determine the optimal number of clusters to use by calculating either cost or inertia of different k,
        running the elbow method and gathering other metrics. Different K are then ranked and the best performing K is
        selected and set as an attribute on self.
        :param k_min: Int. Smallest number of clusters (default 3)
        :param k_max: Int. Largest number of clusters (default 9)
        :param k_increment: Int. How much to increment by per iteration (default 1)
        :param method: Str. Clustering method to be used
        :return: (only if method is kmeans and pca) The best performing algorithm based on n_components of PCA
        """
        if not self.method_kprototypes_or_kmodes:
            self.kprototypes_or_kmodes()

        if not method:
            method = self.method_kprototypes_or_kmodes
        if not isinstance(cluster_df, pd.DataFrame):
            if self.cluster_vars:
                cluster_df = self.data_encoded[self.cluster_vars]
            else:
                cluster_df = self.data_encoded
            cluster_df = cluster_df.loc[:, ~cluster_df.columns.duplicated()].copy()
        # Calc cost fn
        inputs = tqdm(list(range(self.min_k, self.max_k, k_increment)))

        if method == "kprototypes":
            parallel_processed = Parallel(n_jobs=self.num_cores)(
                delayed(self.kprototypes_cost)(i, cluster_df=cluster_df, verbose_binary=1)
                for i in inputs
            )
        elif method == "kmodes":
            parallel_processed = Parallel(n_jobs=self.num_cores)(
                delayed(self.kmodes_cost)(i, cluster_df=cluster_df, verbose_binary=1)
                for i in inputs
            )
        elif method == 'kmeans':
            parallel_processed = Parallel(n_jobs=self.num_cores)(
                delayed(self.kmeans_inertia)(i, cluster_df=cluster_df, verbose_binary=1)
                for i in inputs
            )
        elif method == 'kmeans_and_pca':
            parallel_processed = []
            for i in inputs:
                print(i)
                parallel_processed.append(
                    self.kmeans_and_pca_inertia(i, cluster_df=cluster_df, verbose_binary=1, n_components=None))
            # parallel_processed = Parallel(n_jobs=self.num_cores)(
            #     delayed(self.kmeans_and_pca_inertia)(i, cluster_df=cluster_df, verbose_binary=1, n_components=None)
            #     for i in inputs
            # )
            n_components = [x[2] for x in parallel_processed]
        if all(v is None for v in parallel_processed):
            self.opt_clusters = np.nan
            return

        # Find the elbow
        cost = [x[0] for x in parallel_processed]
        metrics = [x[1] for x in parallel_processed]
        ranked_metrics = rank_cluster_metrics(pd.DataFrame(metrics))

        n_points = len(cost)
        all_coord = np.vstack((range(n_points), cost)).T
        #     np.array([range(n_points), cost])
        first_point = all_coord[0]
        line_vec = all_coord[-1] - all_coord[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
        vec_from_first = all_coord - first_point
        scalar_product = np.sum(
            vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1
        )
        vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
        vec_to_line = vec_from_first - vec_from_first_parallel
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
        idx_of_best_point = np.argmax(dist_to_line)

        y = np.array([i for i in range(self.min_k, self.max_k, 1)])
        # optimal k
        optimal_elbow = y[idx_of_best_point]
        # add top marks to best elbow
        row_of_best_elbow = ranked_metrics.loc[ranked_metrics['n_clusters'] == optimal_elbow].index[0]
        ranked_metrics.at[row_of_best_elbow, 'rank_sum'] += n_points
        # Get best ranked n_clusters as optimal
        ranked_metrics.set_index('n_clusters', inplace=True)
        if hierarchical:
            return ranked_metrics['rank_sum'].idxmax()
        self.opt_clusters = ranked_metrics['rank_sum'].idxmax()
        if method == 'kmeans_and_pca':
            return n_components[row_of_best_elbow]

    @staticmethod
    def kproto_clustering(num_clusters, df_matrix, cols_cat_index, cluster_vars, verbose_binary, seed):
        """
        Static method to fit kprototypes clustering to data
        :param num_clusters: Int. Number of clusters to use
        :param df_matrix: np.Array of cleaned, preprocessed and encoded questionnaire responses
        :param cols_cat_index: A list of categorical columns
        :param cluster_vars: A list of columns to be included in clustering
        :param verbose_binary: Int. 1 if verbose
        :param seed: Int. Seed to be used in random_state
        :return: tuple. Model, cluster centres, cluster cost, labels, and metrics
        """
        kproto_cao = KPrototypes(
            n_clusters=num_clusters, init="Cao", n_init=1, verbose=verbose_binary, n_jobs=-1, random_state=seed
        )

        try:
            cluster_model = kproto_cao.fit(df_matrix, categorical=cols_cat_index)
        except ValueError:
            return

        cluster_cost = kproto_cao.cost_
        raw_cluster_centroids = pd.DataFrame(
            kproto_cao.cluster_centroids_, columns=cluster_vars
        )
        metrics = get_cluster_metrics(
            data=df_matrix,
            cluster_labels=kproto_cao.labels_,
            n_clusters=num_clusters,
            n_seed=seed,
            cols_cat=cols_cat_index)
        return cluster_model, raw_cluster_centroids, cluster_cost, kproto_cao.labels_, metrics

    def kmodes_clustering(self, num_clusters, df_matrix, cluster_vars, verbose_binary, seed):
        """
        Method to fit kmodes clustering to data
        :param num_clusters: Int. Number of clusters to use
        :param df_matrix: np.Array of cleaned, preprocessed and encoded questionnaire responses
        :param cluster_vars: A list of columns to be included in clustering
        :param verbose_binary: Int. 1 if verbose
        :param seed: Int. Seed to be used in random_state
        :return: tuple. Model, cluster centres, cluster cost, labels, and metrics
        """
        kmodes = KModes(
            n_clusters=num_clusters, verbose=verbose_binary, n_jobs=-1, random_state=seed
        )
        cluster_model = kmodes.fit(df_matrix)
        cluster_cost = kmodes.cost_
        raw_cluster_centroids = pd.DataFrame(
            kmodes.cluster_centroids_, columns=cluster_vars
        )
        cluster_labels_ = kmodes.labels_
        if cluster_labels_ is None:
            return None, None, None, None, None
        metrics = get_cluster_metrics(
            data=df_matrix,
            cluster_labels=cluster_labels_,
            n_clusters=num_clusters,
            n_seed=seed)
        return cluster_model, raw_cluster_centroids, cluster_cost, cluster_labels_, metrics

    @staticmethod
    def kmeans_clustering(num_clusters, df_matrix, cluster_vars, verbose_binary, seed):
        """
        Static method to fit kmeans clustering to data
        :param num_clusters: Int. Number of clusters to use
        :param df_matrix: np.Array of cleaned, preprocessed and encoded questionnaire responses
        :param cluster_vars: A list of columns to be included in clustering
        :param verbose_binary: Int. 1 if verbose
        :param seed: Int. Seed to be used in random_state
        :return: tuple. Model, cluster centres, cluster cost, labels, and metrics
        """
        kmeans = KMeans(n_clusters=num_clusters, verbose=verbose_binary)
        cluster_model = kmeans.fit(df_matrix)
        cluster_cost = kmeans.inertia_
        if cluster_vars == ['All']:
            raw_cluster_centroids = pd.DataFrame(
                kmeans.cluster_centers_
            )
        else:
            raw_cluster_centroids = pd.DataFrame(
                kmeans.cluster_centers_, columns=cluster_vars
            )
        cluster_labels_ = kmeans.labels_
        metrics = get_cluster_metrics(df_matrix, cluster_labels_, num_clusters, seed)
        return cluster_model, raw_cluster_centroids, cluster_cost, cluster_labels_, metrics

    def process_hierarchical(self, hierarchical_dict, model):
        labels = []
        combined_df = pd.DataFrame()
        for cluster, data in hierarchical_dict.items():
            if self.ignore_hierarchical_value and cluster in self.ignore_hierarchical_value:
                cluster_labels = [-99] * data[0].shape[0]
            else:
                try:
                    max_labels = max(labels)
                except ValueError:
                    max_labels = -1
                if not isinstance(data[1], pd.Series):
                    n = max_labels + 1
                    labels.append(n)
                    cluster_labels = [n] * data[0].shape[0]
                else:
                    cluster_labels = [x + max_labels + 1 for x in data[1]]
                    labels.append(max(cluster_labels))
            ids = data[0]['alchemer_id']
            labels_df = pd.DataFrame({'alchemer_id': ids, 'cluster': cluster_labels})
            combined_df = pd.concat([combined_df, labels_df], ignore_index=True)
        df_with_clusters = self.data_encoded.merge(combined_df, on='alchemer_id', how='left')
        clusters_in_the_right_order = df_with_clusters['cluster']
        if self.ignore_hierarchical_value:
            df_encoded_not_ignored = df_with_clusters[df_with_clusters['cluster'] != -99]
            df_encoded_not_ignored.drop('cluster', inplace=True, axis=1)
            df_not_ignored = self.data.copy()
            if isinstance(clusters_in_the_right_order, pd.Series):
                clusters_in_the_right_order = clusters_in_the_right_order.to_list()
            df_not_ignored['cluster'] = clusters_in_the_right_order
            df_not_ignored = df_not_ignored[df_not_ignored['cluster'] != -99]
            df_not_ignored.drop('cluster', inplace=True, axis=1)
            clusters = list(filter(lambda x: x != -99, clusters_in_the_right_order))
            chi2_data = self.get_chisquare_data(clusters, clustered_df=df_not_ignored, weights=self.weight_column)
            metrics = get_all_metrics(df_encoded_not_ignored[self.cluster_vars], clusters, max(labels),
                                      None, None, model, full_data=df_not_ignored,
                                      data_encoded=df_encoded_not_ignored, chi2_data=chi2_data)
            return {'model': model,
                    'labels': clusters_in_the_right_order,
                    'metrics': metrics,
                    'inference_data': chi2_data}

        chi2_data = self.get_chisquare_data(clusters_in_the_right_order, weights=self.weight_column)
        metrics = get_all_metrics(self.data_encoded[self.cluster_vars], clusters_in_the_right_order, max(labels) + 1,
                                  None, None, model, full_data=self.full_data,
                                  data_encoded=self.full_data_encoded, chi2_data=chi2_data)
        return {'model': model,
                'labels': clusters_in_the_right_order,
                'metrics': metrics,
                'inference_data': chi2_data}

    def hierarchical_k_clustering(self):
        hierarchical_dict = {}
        for cluster, cluster_df in self.hierarchical_grouped_by:
            opt_k = self.optimal_clusters(cluster_df=cluster_df, hierarchical=True)
            cluster_dict = \
                self.execute_k_clustering(output='hierarchical', hierarchical=cluster_df[self.cluster_vars], k=opt_k)
            hierarchical_dict[cluster] = [cluster_df, cluster_dict['labels']]
        return self.process_hierarchical(hierarchical_dict, f'Hierarchical K using {self.hierarchical}')

    def execute_k_clustering(self, verbose_binary=0, output="all", hierarchical=None, k=None):
        """
        Method to prepare and carry out kprototypes, kmodes, or kmeans clustering using optimal k clusters
        :param verbose_binary: Int. 0 for not verbose, 1 for verbose (default 0)
        :param output: str. Either "all" or "cost". If "cost" then only cost is returned, else all outputs returned
        :return: Dict. Model, labels, cluster centres, cost, and metrics or cost
        """
        if isinstance(hierarchical, pd.DataFrame):
            cluster_df = hierarchical.copy()
        else:
            if self.cluster_vars:
                cluster_df = self.data_encoded[self.cluster_vars]
            else:
                cluster_df = self.data_encoded
            cluster_df = cluster_df.loc[:, ~cluster_df.columns.duplicated()].copy()
        cluster_vars = [x for x in cluster_df.columns if x in self.cluster_vars]
        if not self.method_kprototypes_or_kmodes:
            self.kprototypes_or_kmodes()
        if not self.opt_clusters:
            self.optimal_clusters()
        try:
            self.opt_clusters
        except AttributeError:
            self.optimal_clusters()
        if np.isnan(self.opt_clusters) and not k:
            return {'model': 'kprototypes/kmodes',
                    'labels': np.nan,
                    'cluster_centres': np.nan,
                    'cost': np.nan,
                    'metrics': {'algorithm': 'kprototypes/kmodes',
                                'n_clusters': 'Unable to perform clustering due to initialisation errors'},
                    'inference_data': np.nan}
        elif not k:
            k = self.opt_clusters

        method = self.method_kprototypes_or_kmodes

        df_matrix = cluster_df.to_numpy()
        if method == 'kprototypes':
            cols_cat_index = [cluster_df.columns.get_loc(c) for c in self.categorical_cols]
            clusters = Parallel(n_jobs=self.num_cores)(
                delayed(self.kproto_clustering)(k, df_matrix, cols_cat_index, cluster_vars,
                                                verbose_binary, i)
                for i in self.seeds
            )
        elif method == 'kmodes':
            clusters = Parallel(n_jobs=self.num_cores)(
                delayed(self.kmodes_clustering)(k, df_matrix, cluster_vars, verbose_binary, i)
                for i in self.seeds
            )
        elif method == 'kmeans':
            clusters = Parallel(n_jobs=self.num_cores)(
                delayed(self.kmeans_clustering)(k, df_matrix, self.cluster_vars, verbose_binary, i)
                for i in self.seeds
            )
        clusters = list(filter(None, clusters))
        if not clusters:
            metrics = {'algorithm': method, 'n_clusters': 'Unable to initialise for all k'}
            return {'model': None, 'labels': None, 'metrics': metrics}

        metrics_from_clusters = [x[4] for x in clusters]
        metrics_from_clusters = self.remove_unbalanced_clustering(pd.DataFrame(metrics_from_clusters))
        if metrics_from_clusters.empty:
            metrics = {'algorithm': method, 'n_clusters': 'Unable to perform clustering due to unbalanced clusters'
                                                          ' for all k'}
            return {'model': None, 'labels': None, 'metrics': metrics}
        ranked_metrics = rank_cluster_metrics(metrics_from_clusters)
        best_seed_clustering = ranked_metrics['rank_sum'].idxmax()
        cluster_model, raw_cluster_centroids, cluster_cost, cluster_labels_, cluster_metrics = clusters[
            best_seed_clustering]
        if hierarchical:
            return {'model': cluster_model, 'labels': cluster_labels_}
        cluster_centroids = raw_cluster_centroids.T
        chi2_data = self.get_chisquare_data(cluster_labels_, weights=self.weight_column)
        metrics = get_all_metrics(cluster_df, cluster_labels_, self.opt_clusters, cluster_model,
                                  cluster_model, method, cluster_metrics['n_seed'],
                                  self.categorical_cols, chi2_data=chi2_data, full_data=self.full_data,
                                  data_encoded=self.full_data_encoded.copy(deep=True))
        if output == "all":
            return {'model': cluster_model,
                    'labels': cluster_labels_,
                    'cluster_centres': cluster_centroids,
                    'cost': cluster_cost,
                    'metrics': metrics,
                    'inference_data': chi2_data}
        elif output == "cost":
            return cluster_cost

    def hierarchical_kmeans_and_pca(self):
        hierarchical_dict = {}
        for cluster, cluster_df in self.hierarchical_grouped_by:
            cluster_dict = \
                self.kmeans_and_pca_clustering(hierarchical=cluster_df[self.cluster_vars])
            hierarchical_dict[cluster] = [cluster_df, cluster_dict['labels']]
        return self.process_hierarchical(hierarchical_dict, f'Hierarchical Kmeans using {self.hierarchical}')

    def kmeans_and_pca_clustering(self, hierarchical=None, mode='eval'):
        """
        Method to prepare and execute kmeans and pca clustering with optimal k and optimal n_components
        :param return_chi_2_data: Bool. **Only used for testing** returns the chi_2_data object in return dict. Default
            False
        :return: Dict. Model, labels, cluster centres, intertia, and metrics
        """
        n_components = self.optimal_clusters(method='kmeans_and_pca')
        if isinstance(hierarchical, pd.DataFrame):
            df, scaler = self.standardize_data(hierarchical)
        else:
            if self.cluster_vars:
                df, scaler = self.standardize_data(self.data_encoded[self.cluster_vars])
            else:
                df, scaler = self.standardize_data(self.data_encoded)
        try:
            df_matrix = df.to_numpy()
        except AttributeError:
            df_matrix = df
        if not self.opt_clusters:
            self.optimal_clusters()
        kmeans_and_metrics = Parallel(n_jobs=self.num_cores)(
            delayed(self.get_pca_and_kmeans)(df_matrix, self.opt_clusters, n_components, i)
            for i in self.seeds)
        metrics_from_clusters = [x[1] for x in kmeans_and_metrics]
        metrics_from_clusters = self.remove_unbalanced_clustering(pd.DataFrame(metrics_from_clusters))
        if metrics_from_clusters.empty:
            metrics = {'algorithm': 'Kmeans & PCA',
                       'n_clusters': 'Unable to perform clustering due to unbalanced clusters'
                                     ' for all k'}
            return {'model': None, 'labels': None, 'metrics': metrics}
        ranked_metrics = rank_cluster_metrics(metrics_from_clusters)
        best_seed_clustering = ranked_metrics['rank_sum'].idxmax()
        optimal_kmeans, metrics, pca_data, pca = kmeans_and_metrics[best_seed_clustering]
        if isinstance(hierarchical, pd.DataFrame):
            return {'model': optimal_kmeans, 'labels': optimal_kmeans.labels_}
        raw_cluster_centers = pd.DataFrame(
            optimal_kmeans.cluster_centers_
        )

        if mode == 'tune':
            metrics = get_cluster_metrics(pca_data, optimal_kmeans.labels_, metrics['n_clusters'],
                                          n_seed=metrics['n_seed']
                                          , full_data=self.data)
            clustering_dict = {'model': optimal_kmeans,
                               'labels': optimal_kmeans.labels_,
                               'cluster_centres': raw_cluster_centers,
                               'cost': optimal_kmeans.inertia_,
                               'metrics': metrics}
        elif mode == 'eval':
            chi2_data = self.get_chisquare_data(optimal_kmeans.labels_, weights=self.weight_column)
            metrics = get_all_metrics(pca_data, optimal_kmeans.labels_, metrics['n_clusters'], optimal_kmeans,
                                      clone(optimal_kmeans), 'PCA and Kmeans', n_seed=metrics['n_seed'],
                                      chi2_data=chi2_data, full_data=self.full_data,
                                      data_encoded=self.full_data_encoded)

            clustering_dict = {'model': optimal_kmeans,
                               'labels': optimal_kmeans.labels_,
                               'cluster_centres': raw_cluster_centers,
                               'cost': optimal_kmeans.inertia_,
                               'inference_data': chi2_data,
                               'metrics': metrics,
                               'pca_model': pca,
                               'scaler_model': scaler}
        else:
            raise ValueError('Allowed values for mode are either "tune" or "eval".')

        return clustering_dict

    def remove_unbalanced_clustering(self, metrics_df):
        upper = self.upper_cluster_threshold
        lower = self.lower_cluster_threshold
        if isinstance(metrics_df, dict):
            if all(lower < v < upper for v in metrics_df['cluster_proportions']):
                return metrics_df
            else:
                return pd.DataFrame()
        else:
            try:
                print(metrics_df['cluster_proportions'])
                mask = metrics_df.apply(lambda x: True if all(lower < v <
                                                              upper for v in
                                                              x['cluster_proportions'].values()) else False, axis=1)
                removed_unbalanced = metrics_df[mask]
            except KeyError:
                return pd.DataFrame()
            return removed_unbalanced

    def hierarchical_lca(self):
        hierarchical_dict = {}
        for cluster, cluster_df in self.hierarchical_grouped_by:
            cluster_dict = \
                self.lca_with_timer(hierarchical=cluster_df[self.cluster_vars])
            hierarchical_dict[cluster] = [cluster_df, cluster_dict['class_labels']]
        return self.process_hierarchical(hierarchical_dict, f'Hierarchical LCA using {self.hierarchical}')

    def reencode_agree_not_agree(self, cluster_df):
        reencoding_dict = {
            'Agree': 1,
            'Strongly agree': 1,
            'Neither agree nor disagree': 0,
            'Neither agree not disagree': 0,
            'Disagree': 0,
            'Strongly disagree': 0
        }
        for col in self.cluster_vars:
            text_responses = self.data[col].value_counts()
            if 'Agree' in text_responses.index or 'agree' in text_responses.index:
                cluster_df[col] = self.data[col].replace(reencoding_dict)
        return cluster_df

    def lca_with_timer(self, hierarchical=None):
        """
        Method to call R scripts to perform Latent Class Analysis or Mixture modelling for latent classes. If this takes
        longer than time declared in the lca() function then it will exit with an RRuntimeError.
        :return: Dict. Model string, labels, and metrics or Error
        """

        if isinstance(hierarchical, pd.DataFrame):
            cluster_df = hierarchical.copy()
        else:
            cluster_df = self.data_encoded.copy()
        cluster_df = self.reencode_agree_not_agree(cluster_df)
        if not self.method_kprototypes_or_kmodes:
            self.kprototypes_or_kmodes()

        r = robjects.r
        try:
            r['source']('../app/lca.R')
        except embedded.RRuntimeError:
            r['source']('app/lca.R')
        lca_r = robjects.globalenv['lca']
        mixture_model_r = robjects.globalenv['mixture_modelling_lca']

        if self.method_kprototypes_or_kmodes == 'kmodes':
            with localconverter(robjects.default_converter + pandas2ri.converter):
                data_for_r = robjects.conversion.py2rpy(cluster_df[self.cluster_vars])
                lca_result = lca_r(data_for_r, self.cluster_vars)
                lca_result['class_labels'] = list(lca_result['class_labels'])
        else:
            datatypes = ['categorical' if x in self.categorical_cols else 'continuous' for x in
                         cluster_df[self.cluster_vars].columns]
            with localconverter(robjects.default_converter + pandas2ri.converter):
                data_for_r = robjects.conversion.py2rpy(cluster_df[self.cluster_vars])
                lca_result = mixture_model_r(data_for_r, self.cluster_vars, datatypes)
        if isinstance(hierarchical, pd.DataFrame):
            return lca_result
        n_clusters = len(set(lca_result['class_labels']))
        chi2_data = self.get_chisquare_data(lca_result['class_labels'], weights=self.weight_column)
        if not isinstance(chi2_data[0], pd.DataFrame):
            return {'model': 'poLCA model', 'labels': 'Unable to complete due to unbalanced clusters'}
        metrics = get_all_metrics(cluster_df[self.cluster_vars], lca_result['class_labels'],
                                  n_clusters, None, None, chi2_data=chi2_data, algo='poLCA', n_seed=99,
                                  full_data=self.full_data, data_encoded=self.full_data_encoded.copy(deep=True))
        return {'model': 'poLCA model', 'labels': lca_result['class_labels'],
                'metrics': metrics, 'inference_data': chi2_data}

    def lca(self, test_wait_time=False):
        """
        Method to call lca_with_timer() function with a timer wrapper to ensure it does not run for a very long time.
        This is an issue with some mixed models in R as the complexity of the data can lead to very long run times
        (upwards of 4 hours) when the data are not suited to this kind of analysis.
        :param test_wait_time: Bool (Optional). Sets timeout timer to 2 seconds if true (used in testing only)
        :return: Dict. Model string, labels, and metrics or dict of strings denoting timeout error
        """
        if test_wait_time:
            timer = 2
        else:
            timer = 3000

        @exit_after(timer)
        def pass_to_lca():
            try:
                lca_return_obj = self.lca_with_timer()
            except embedded.RRuntimeError as e:
                print(e)
                lca_return_obj = {'model': 'poLCA model', 'labels': 'lca timed out', 'metrics': {'algorithm': 'lca timed out'}}
            return lca_return_obj

        lca_return_obj = pass_to_lca()

        return lca_return_obj

    @staticmethod
    def parallel_bmm(data, n_classes, seed):
        # fit a Bayesian Gaussian mixture model to the data
        try:
            model = BayesianGaussianMixture(n_components=n_classes, random_state=seed)
            model.fit(data)
        except ValueError:
            try:
                model = BayesianGaussianMixture(n_components=n_classes, reg_covar=1e-5, random_state=seed)
                model.fit(data)
            except ValueError:
                return

        # compute the probability that each individual belongs to each latent class
        probs = model.predict_proba(data)

        def get_n_parameters(model, X):
            _, n_features = model.means_.shape
            n_effective_components = len(np.unique(model.predict(X)))
            if model.covariance_type == 'full':
                cov_params = n_effective_components * n_features * (n_features + 1) / 2.
            elif model.covariance_type == 'diag':
                cov_params = n_effective_components * n_features
            elif model.covariance_type == 'tied':
                cov_params = n_features * (n_features + 1) / 2.
            elif model.covariance_type == 'spherical':
                cov_params = n_effective_components
            mean_params = n_features * n_effective_components
            return int(cov_params + mean_params + n_effective_components - 1)

        score = model.score(data)
        n_params = get_n_parameters(model, data)

        bic = (-2 * score * data.shape[0] + n_params * np.log(data.shape[0]))

        aic = -2 * score * data.shape[0] + 2 * n_params
        probs_df = pd.DataFrame(probs, columns=[i for i in range(n_classes)])
        labels = probs_df.idxmax(1)
        if min(labels.value_counts()) < data.shape[0] * 0.01:
            return
        metrics = get_cluster_metrics(data, labels, n_classes, n_seed=seed)
        metrics['bic'] = bic
        metrics['aic'] = aic
        # create a dataframe of the results
        return labels, model, metrics

    def baysian_mixture_modelling(self, data, n_classes):
        """
        Perform latent class analysis on a dataset with categorical variables.

        Parameters
        ----------
        data : pandas DataFrame
            The data to be analyzed, with rows representing individuals and columns representing variables.
        n_classes : int
            The number of latent classes to estimate.

        Returns
        -------
        pandas DataFrame
            A dataframe containing the probability that each individual belongs to each latent class.
        """
        bmm_models = Parallel(n_jobs=self.num_cores)(
            delayed(self.parallel_bmm)(data, n_classes, i)
            for i in self.seeds)

        metrics_from_clusters = []
        for bmm in bmm_models:
            try:
                metrics_from_clusters.append(bmm[2])
            except TypeError:
                pass
        metrics_from_clusters = self.remove_unbalanced_clustering(pd.DataFrame(metrics_from_clusters))
        if metrics_from_clusters.empty:
            return None
        ranked_metrics = rank_cluster_metrics(metrics_from_clusters, information_criterions=True)
        best_seed_clustering = ranked_metrics['rank_sum'].idxmax()
        results = bmm_models[best_seed_clustering]
        return results

    def hierarchical_bmm(self):
        hierarchical_dict = {}
        for cluster, cluster_df in self.hierarchical_grouped_by:
            cluster_dict = \
                self.bmm(hierarchical=cluster_df[self.cluster_vars])
            hierarchical_dict[cluster] = [cluster_df, cluster_dict['labels']]
        return self.process_hierarchical(hierarchical_dict, f'Hierarchical BMM using {self.hierarchical}')

    def bmm(self, hierarchical=None):
        if isinstance(hierarchical, pd.DataFrame):
            cluster_df = hierarchical.copy()
        else:
            cluster_df = self.full_data_encoded[self.cluster_vars]
        bmm = Parallel(n_jobs=self.num_cores)(
            delayed(self.baysian_mixture_modelling)(cluster_df, i)
            for i in range(3, 9))
        bmm = list(filter(None, bmm))
        if not bmm:
            metrics = {'algorithm': 'BMM', 'n_clusters': 'very small clusters detected'}
            return {'model': None, 'labels': None, 'metrics': metrics}
        returned_metrics = [x[2] for x in bmm]
        returned_metrics = self.remove_unbalanced_clustering(pd.DataFrame(returned_metrics))
        if returned_metrics.empty:
            metrics = {'algorithm': 'baysian mixture modelling',
                       'n_clusters': 'Unable to perform clustering due to unbalanced clusters'
                                     ' for all k'}
            return {'model': None, 'labels': None, 'metrics': metrics}
        ranked_metrics = rank_cluster_metrics(pd.DataFrame(returned_metrics), information_criterions=True)
        best_clustering = ranked_metrics['rank_sum'].idxmax()
        best_bmm_labels = bmm[best_clustering][0]
        best_bmm_model = bmm[best_clustering][1]
        if isinstance(hierarchical, pd.DataFrame):
            return {'model': best_bmm_model, 'labels': best_bmm_labels}
        chi2_data = self.get_chisquare_data(best_bmm_labels, weights=self.weight_column)
        metrics = get_all_metrics(self.full_data_encoded[self.cluster_vars], best_bmm_labels,
                                  best_bmm_model.n_components, model=best_bmm_model, fitted_model=best_bmm_model,
                                  algo='Baysian Mixture Modelling', chi2_data=chi2_data, full_data=self.full_data,
                                  data_encoded=self.full_data_encoded.copy(deep=True),
                                  n_seed=best_bmm_model.random_state)
        return {'model': best_bmm_model, 'labels': best_bmm_labels, 'metrics': metrics,
                'inference_data': chi2_data}

    @staticmethod
    def drop_rules_based_columns(df, q_codes):
        for q in q_codes:
            try:
                df.drop(q, axis=1, inplace=True)
            except KeyError:
                pass
        return df

    def rules_based(self, q_codes):
        """
        Segment based on a single column in the dataset as defined by the research team
        Parameters
        ----------
        q_code str: Name of column

        Returns dict: Contains model, labels, metrics, and inference data
        -------
        """
        return_objects = []
        for q_code in q_codes:
            full_data_copy = self.full_data.copy()
            full_data_encoded_copy = self.full_data_encoded.copy()
            data_encoded_copy = self.full_data_encoded.copy()
            data_encoded_copy = self.drop_rules_based_columns(data_encoded_copy, q_codes)
            full_data_copy = self.drop_rules_based_columns(full_data_copy, q_codes)
            full_data_encoded_copy = self.drop_rules_based_columns(full_data_encoded_copy, q_codes)
            cluster_column_unencoded = self.full_data[q_code].copy()
            print(cluster_column_unencoded)
            cluster_labels = LabelEncoder().fit_transform(self.full_data[q_code])
            chi2_data = self.get_chisquare_data(cluster_labels, segmentation=q_code, rename_segments=cluster_column_unencoded,
                                                clustered_df=full_data_copy, weights=self.weight_column)
            metrics = get_all_metrics(data_encoded_copy[self.cluster_vars], cluster_labels,
                                      len(np.unique(cluster_labels)),
                                      model=None, fitted_model=None, algo=f'rules-based - {q_code}',
                                      chi2_data=chi2_data,
                                      full_data=full_data_copy, data_encoded=full_data_encoded_copy.copy(deep=True))
            return_objects.append({'model': f'Rules-Based - {q_code}', 'labels': cluster_labels, 'metrics': metrics,
                                   'inference_data': chi2_data})
        # return_objects = Parallel(n_jobs=len(q_codes))(
        #     delayed(self.parallel_rules_based)(i, q_codes=q_codes)
        #     for i in q_codes
        # )

        if self.cluster_vars:
            try:
                self.cluster_vars.remove(q_codes)
            except (ValueError, KeyError) as e:
                pass
        return return_objects

    def run_all_segmentations(self, q_code=None):
        if q_code:
            seg_list = ['kmeans', 'bmm', 'kprototypes', 'rules_based']
        else:
            seg_list = ['kmeans', 'bmm', 'kprototypes']

        parallel_segmentations = Parallel(n_jobs=len(seg_list))(
            delayed(self.parallel_run_segmentations)(i, q_codes=q_code)
            for i in seg_list
        )

        segmentations = {}

        for segmentation in parallel_segmentations:
            segmentations[list(segmentation.keys())[0]] = list(segmentation.values())[0]


        # segmentations = {}
        # logger.info('Started segmentations')
        # if self.hierarchical:
        #     return self.get_hierarchical_segmentations(q_codes=q_code)
        # start = time.time()
        # if q_code:
        #     if isinstance(q_code, str):
        #         q_code = [q_code]
        #     rules_based = self.rules_based(q_code)
        #     for i, q in enumerate(q_code):
        #         segmentations[f'{q}'] = rules_based[i]
        #     self.data.drop(q_code, inplace=True, axis=1)
        #     self.data_encoded.drop(q_code, inplace=True, axis=1)
        #     self.full_data.drop(q_code, inplace=True, axis=1)
        #     self.full_data_encoded.drop(q_code, inplace=True, axis=1)
        #     recommended_finish = time.time()
        #     logger.info(f'Recommended segmentation completed in {round(recommended_finish - start, 1)}')
        #
        # segmentations['baysian_mixture_modelling'] = self.bmm()
        # bmm_finish = time.time()
        # try:
        #     logger.info(f'BMM segmentation completed in {round(bmm_finish - recommended_finish, 1)}')
        # except UnboundLocalError:
        #     logger.info(f'BMM segmentation completed in {round(bmm_finish - start, 1)}')
        lca = self.lca()

        if isinstance(lca['metrics'], str):
            error_str = lca['metrics']
            lca['metrics'] = {'algorithm': 'Latent Class Analysis',
                              'n_seed': 'Unable to complete LCA',
                              'n_clusters': error_str}
        segmentations['latent_class_analysis'] = lca
        # lca_finish = time.time()
        # logger.info(f'LCA segmentation completed in {round(lca_finish - bmm_finish, 1)}')
        # segmentations['kmeans'] = self.kmeans_and_pca_clustering()
        # kmeans_finish = time.time()
        # logger.info(f'Kmeans + PCA segmentation completed in {round(kmeans_finish - lca_finish, 1)}')
        #
        # segmentations['kprototypes'] = self.execute_k_clustering()
        # kproto_finish = time.time()
        # logger.info(f'Kprototypes/Kmodes segmentation completed in {round(kproto_finish - kmeans_finish, 1)}')
        # logger.info(f'All segmentations completed in {round(time.time() - start, 1)} seconds')
        return segmentations

    def parallel_rules_based(self, q_code, q_codes):
        full_data_copy = self.full_data.copy()
        full_data_encoded_copy = self.full_data_encoded.copy()
        data_encoded_copy = self.full_data_encoded.copy()
        data_encoded_copy = self.drop_rules_based_columns(data_encoded_copy, q_codes)
        full_data_copy = self.drop_rules_based_columns(full_data_copy, q_codes)
        full_data_encoded_copy = self.drop_rules_based_columns(full_data_encoded_copy, q_codes)
        cluster_column_unencoded = self.full_data[q_code].copy()
        print(cluster_column_unencoded)
        cluster_labels = LabelEncoder().fit_transform(self.full_data[q_code])
        chi2_data = self.get_chisquare_data(cluster_labels, segmentation=q_code,
                                            rename_segments=cluster_column_unencoded,
                                            clustered_df=full_data_copy, weights=self.weight_column)
        metrics = get_all_metrics(data_encoded_copy[self.cluster_vars], cluster_labels,
                                  len(np.unique(cluster_labels)),
                                  model=None, fitted_model=None, algo=f'rules-based - {q_code}',
                                  chi2_data=chi2_data,
                                  full_data=full_data_copy, data_encoded=full_data_encoded_copy.copy(deep=True))
        return {'model': f'Rules-Based - {q_code}', 'labels': cluster_labels, 'metrics': metrics,
                               'inference_data': chi2_data}



    def get_hierarchical_segmentations(self, q_codes=None):
        start = time.time()
        segmentations = {}
        if q_codes:
            if isinstance(q_codes, str):
                q_codes = [q_codes]
            for q in q_codes:
                segmentations[f'recommended - {q}'] = self.rules_based(q, q_codes)
                recommended_finish = time.time()
                logger.info(f'Recommended segmentation completed in {round(recommended_finish - start, 1)}')
        segmentations['kmeans'] = self.hierarchical_kmeans_and_pca()
        segmentations['lca'] = self.hierarchical_lca()
        segmentations['baysian_mixture_modelling'] = self.hierarchical_bmm()
        segmentations['kprototypes'] = self.hierarchical_k_clustering()
        return segmentations

    def parallel_run_segmentations(self, seg_algorithm, q_codes=None):
        seg_mapping = {
            'kmeans': self.kmeans_and_pca_clustering,
            'bmm': self.bmm,
            'kprototypes': self.execute_k_clustering,
            'rules_based': self.rules_based
        }
        segmentation = seg_mapping[seg_algorithm](q_codes)
        return {seg_algorithm: segmentation}
