import logging
import re
import time
import os
import numpy as np
import pandas as pd
import s3fs
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats.contingency import expected_freq
try:
    from kraken.app.inference.helper_functions import remove_not_selected, remove_not_select_and_cat_percentage
except ModuleNotFoundError:
    from kraken.app.inference.helper_functions import remove_not_selected, remove_not_select_and_cat_percentage

log = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
log.setLevel(logging.DEBUG)

s3_check = s3fs.S3FileSystem()


def strip_html(x):
    """Function to strip html tags from the string"""
    x = str(x)
    try:
        return re.sub('<[^<]+?>', '', x)
    except:
        raise ValueError(f'Inspect regex for {x}')


class ChiSquaredTester:
    """
    A class for performing chi-squared contingency table statistics on variables among different segments or clusters.
    It determines if there is a significant difference between the observed frequency and expected frequency counts
    among segments or clusters.

    Parameters
    ----------
    survey_name : str
        The name of the survey or dataset.
    segmentation : str or None
        A description or label for segmentation (optional).
    clustered_data : pd.DataFrame
        The response data (processed and cleaned).
    seg_col : str
        The column header which denotes the segment or cluster labels.
    rename_segments : pd.Series or None
        A Series mapping cluster labels to custom names (for renaming).
    conf_interval : float, default: 0.95
        The confidence interval at which the chi-squared test should be executed, i.e., alpha.
    weights : str or None
        The name of the column containing weights (if applicable).
    correction : str or None
        The type of correction to apply for the chi-squared test (e.g., "bonferroni").

    Attributes
    ----------
    cluster_results : dict
        A dictionary containing detailed information for each cluster.
    deliver_pg_stats : pd.DataFrame
        Dataframe with necessary variables to populate both the Discover and Deliver page elements.

    Notes
    -----
    Documentation of the chi-squared statistic:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html

    Additional Methods
    ------------------
    flag_not_selected(cluster, q_ref_table, req_shortname):
        Flags and processes data related to questions where "not selected" is a valid response.

    check_pop_modes_exist():
        Checks if population modes data exists for the survey.

    json_sample_results():
        Generates JSON-style sample results based on the survey data.

    """

    def __init__(self, survey_name, segmentation, clustered_data, seg_col, rename_segments, conf_interval, weights,
                 correction):
        """
        Initialize the ChiSquaredTester class with data and parameters.

        Parameters
        ----------
        survey_name : str
            The name of the survey or dataset.
        segmentation : str or None
            A description or label for segmentation (optional).
        clustered_data : pd.DataFrame
            The response data (processed and cleaned).
        seg_col : str
            The column header which denotes the segment or cluster labels.
        rename_segments : pd.Series or None
            A Series mapping cluster labels to custom names (for renaming).
        conf_interval : float, default: 0.95
            The confidence interval at which the chi-squared test should be executed, i.e., alpha.
        weights : str or None
            The name of the column containing weights (if applicable).
        correction : str or None
            The type of correction to apply for the chi-squared test (e.g., "bonferroni").
        """
        self.cluster_col = seg_col
        self.ci = conf_interval
        self.weights = weights
        self.correction = correction
        self.data = self.remove_cint(clustered_data)
        self.cluster_results = None
        self.deliver_pg_stats = None
        self.survey_name = survey_name
        self.segmentation = segmentation
        self.rename_segments = rename_segments
        self.env = self.get_env()
        try:
            if self.rename_segments:
                self.data_labelled = self.data.join(self.rename_segments)
            else:
                self.data_labelled = self.data.copy()
        except ValueError:
            if self.rename_segments.any():
                self.data_labelled = self.data.join(self.rename_segments)
            else:
                self.data_labelled = self.data.copy()

        if self.segmentation and 'qudo_' in self.segmentation:
            if not self.check_pop_modes_exist():
                self.json_sample_results()

        # self.json_sample_results()

    def get_env(self):
        return os.getenv('PIPELINE_ENV', 'test')

    @staticmethod
    def remove_cint(df):
        """
        Removes 'cint' and 'qudo_weight' columns from a DataFrame and fills any missing values with 'not selected'.

        Parameters:
        df (pandas.DataFrame): The DataFrame from which 'cint' and 'qudo_weight' columns should be removed.

        Returns:
        pandas.DataFrame: A modified DataFrame with the specified columns removed and missing values filled with 'not selected'.

        Notes:
        - This method removes columns containing 'cint' or 'qudo_weight' in their names.
        - Any missing values in the DataFrame are replaced with 'not selected'.
        - If a column is not found, no action is taken, and no exceptions are raised.
        """
        df.fillna('not selected', inplace=True)
        cint_cols = [x for x in df.columns if 'cint' in x]
        weight_columns = [x for x in df.columns if
                          'qudo_weight' in x]  # todo: added it to inference_excluded_cols but leaving it for now
        for col in cint_cols + weight_columns:
            try:
                df.drop(col, inplace=True, axis=1)
            except (ValueError, KeyError) as e:
                pass
        return df

    def inference_excluded_cols(self):
        """
        Identify columns to exclude from inference while allowing them for other usage cases like weighting.

        Returns:
        list of str: A list of column names that should be excluded from inference but can be used for other purposes.

        Notes:
        - This method identifies columns to be excluded from inference but retains them for other use cases.
        - Columns related to segmentation and certain keywords like "weightgain" and "weightwatch" are excluded from inference.
        - If custom weights are specified, they are also added to the list of excluded columns.
        - The returned list of column names can be used to filter out columns during inference processes.
        """
        exception_keywords = ["weightgain", "weightwatch"]
        cols_to_exclude = []
        if self.weights:
            cols_to_exclude.append(self.weights)

        segmentation_cols = [item for item in self.data.columns if "segmentation" in item.lower()]
        remaining_weights = [item for item in self.data.columns if
                             ("weight" in item.lower()) and (not any(ele in item for ele in exception_keywords))]

        cols_to_exclude = cols_to_exclude + segmentation_cols + remaining_weights

        return cols_to_exclude

    def crosstab(self, var):
        """
        Create a cross-tabulation (crosstab) for a single variable across clusters, excluding "Not shown" entries.

        Parameters
        ----------
        var : str
            Any variable within the clustered data.

        Returns
        -------
        tuple
            crosstab_df : pandas.DataFrame
                A cross-tabulation with clusters as columns and variable categories as rows.
                The cells contain observed frequency counts.
            below_threshold : bool
                True if any cell in the crosstab contains a value <= 5; False otherwise.

        Notes
        -----
        This method creates a cross-tabulation of a variable across clusters, excluding entries labeled as "Not shown."

        The resulting crosstab_df contains observed frequency counts.

        Columns with all-zero entries are retained to maintain the structure of the crosstab.

        If any cell in the crosstab contains a value less than or equal to 5, below_threshold is set to True.

        Example
        -------
        crosstab, below_threshold = obj.crosstab('some_variable')
        if below_threshold:
            # Handle the case where the crosstab has cells with low counts.
        else:
            # Continue analysis with the crosstab.
        """

        if "Not shown" in self.data[var].unique().tolist():
            sliced_data = self.data[self.data[var] != "Not shown"].copy(deep=True)
            # todo: this might need to be replaced to still produce a full crosstab but drop columns of 0 entry only
            sliced_data = sliced_data[
                sliced_data[self.cluster_col] != "Not shown"
                ].copy(deep=True)
            # sliced_data.reset_index(inplace=True, drop=True)
            sliced_data = sliced_data.drop(columns=self.inference_excluded_cols())
        else:
            sliced_data = self.data.copy(deep=True)
            sliced_data = sliced_data.drop(columns=self.inference_excluded_cols())

        crosstab_df = pd.crosstab(sliced_data[var], sliced_data[self.cluster_col])
        # todo: think about the zero problem - this is not amended properly yet (variables with crosstab cells below 5
        #  should be dropped
        crosstab_df = crosstab_df.fillna(0.0)
        # crosstab_df = crosstab_df.replace(0, 0.00001)

        # for col in crosstab_df.columns: # OLD IMPLEMENTATION -- DELETE WHEN READY
        #     crosstab_df = crosstab_df[~(crosstab_df[col] < 5)].copy(deep=True)

        # Check if any value is <= 5
        below_threshold = crosstab_df.applymap(lambda x: x <= 5)
        # Check if any value is True in the entire crosstab
        if below_threshold.any().any():
            return crosstab_df, True

        return crosstab_df, False

    @staticmethod
    def crosstab_percent(crosstab_input):
        """
        Create a crosstabulation (crosstab) of percentages for a single variable across clusters within a cluster.

        This method depends on the `crosstab()` class function.

        Parameters
        ----------
        crosstab_input : pandas.DataFrame
            A crosstabulation of observed frequencies.

        Returns
        -------
        pandas.DataFrame
            A crosstab with clusters as columns and variable categories as rows.
            The cells contain the percentage frequency per cluster.

        Notes
        -----
        This method calculates the percentage of each category within a variable for each cluster.
        It depends on the `crosstab()` method for the initial crosstabulation of observed frequencies.

        Example
        -------
        crosstab_df = obj.crosstab('some_variable')
        percent_crosstab = MyClass.crosstab_percent(crosstab_df)
        # Use `percent_crosstab` for further analysis or visualization of percentages.
        """

        for k in crosstab_input.columns:
            crosstab_input[k] = round(
                (crosstab_input[k] / crosstab_input[k].sum()) * 100, 1
            )

        return crosstab_input

    @staticmethod
    def expected_crosstab(crosstab_input):
        """
        Create a crosstabulation (crosstab) of expected frequency counts for a single variable across clusters.

        This method depends on the `crosstab()` class function.

        Parameters
        ----------
        crosstab_input : pandas.DataFrame
            A crosstabulation of observed frequencies.

        Returns
        -------
        pandas.DataFrame
            A crosstab with clusters as columns and variable categories as rows.
            The cells contain the expected frequency counts per cluster.

        Notes
        -----
        This method calculates the expected frequency counts for each category within a variable for each cluster.
        It depends on the `crosstab()` method for the initial crosstabulation of observed frequencies.

        Example
        -------
        crosstab_df = obj.crosstab('some_variable')
        expected_crosstab_df = MyClass.expected_crosstab(crosstab_df)
        # Use `expected_crosstab_df` for statistical analysis or hypothesis testing.
        """
        expected = expected_freq(crosstab_input)  # contingency table
        expected = pd.DataFrame(expected)
        expected.index = crosstab_input.index
        expected.columns = crosstab_input.columns

        return expected

    @staticmethod
    def chi2_stats(crosstab_input, yates=False):
        """
        Perform a chi-square test to check for significant differences between clusters of a given variable (var).

        This method depends on the `crosstab()` class function.

        Parameters
        ----------
        crosstab_input : pandas.DataFrame
            A crosstabulation of observed frequencies.
        yates : bool, optional
            Determines whether Yates correction should be applied for low-frequency counts.

        Returns
        -------
        tuple
            A tuple containing the following statistics:
            - stat (float): Chi-squared statistic.
            - p (float): p-value of the Chi-squared test.
            - dof (int): Degrees of freedom.
            - expected (array): Expected frequency count matrix.

        Notes
        -----
        - This method conducts a chi-square test to assess if there are significant differences between clusters of a variable.
        - It can apply Yates correction if specified by setting `yates=True`.
        - The method depends on the `crosstab()` method for the initial crosstabulation of observed frequencies.
        - In case of any ValueErrors during the test, it returns default values (None, 1.0, None, None).

        Example
        -------
        crosstab_df = obj.crosstab('some_variable')
        chi2_stat, p_value, degrees_of_freedom, expected_counts = MyClass.chi2_stats(crosstab_df, yates=True)
        if p_value < 0.05:
            # Reject the null hypothesis; there are significant differences between clusters.
        else:
            # Fail to reject the null hypothesis; no significant differences found.
        """

        try:
            if yates:
                stat, p, dof, expected = chi2_contingency(crosstab_input, correction=True)
            else:
                stat, p, dof, expected = chi2_contingency(crosstab_input, correction=False)
        except ValueError:
            stat, p, dof, expected = None, 1.0, None, None  # todo: think that over

        return stat, p, dof, expected

    @staticmethod
    def direction(crosstab_input, expected_crosstab):
        """
        Determine if observed values are larger than expected values for a given crosstab.

        This method depends on the `crosstab()` and `expected_crosstab()` class functions.

        Parameters
        ----------
        crosstab_input : pandas.DataFrame
            The observed crosstabulation.
        expected_crosstab : pandas.DataFrame
            The expected frequency counts as a crosstabulation.

        Returns
        -------
        pandas.DataFrame
            A crosstab with clusters as columns and variable categories as rows.
            The cells contain boolean values: True if observed > expected, else False.

        Notes
        -----
        - This method compares observed values to expected values in a crosstabulation.
        - It depends on the `crosstab()` and `expected_crosstab()` methods for obtaining the necessary data.
        - The returned crosstab indicates whether observed values are larger than expected values for each category and cluster.

        Example
        -------
        observed_crosstab = obj.crosstab('some_variable')
        expected_crosstab = MyClass.expected_crosstab(observed_crosstab)
        direction_crosstab = MyClass.direction(observed_crosstab, expected_crosstab)
        # Use `direction_crosstab` to identify clusters where observed values are larger than expected.
        """
        o_minus_e = crosstab_input - expected_crosstab
        o_minus_e = o_minus_e.mask(o_minus_e > 0).isna()

        return o_minus_e

    @staticmethod
    def adjusted_residual(observed_crosstab, expected_crosstab, i, j):
        """
        Calculate the adjusted residual of a value in a contingency table/crosstab for Chi-squared post hoc testing.

        This statistic is used to assess the significance of the difference between an observed value and its expected value.

        Reference: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0188709

        Parameters
        ----------
        observed_crosstab : pandas.DataFrame
            Contingency table with observed frequencies of two variables.
        expected_crosstab : pandas.DataFrame
            Contingency table with expected frequencies of two variables.
        i : int
            Row index of the desired value.
        j : int
            Column index of the desired value.

        Returns
        -------
        float
            The adjusted residual of a crosstab value at location [i, j], rounded to 3 decimal places.

        Notes
        -----
        - The adjusted residual measures the significance of the difference between an observed value and its expected value
        in a contingency table.
        - It is commonly used for Chi-squared post hoc tests.
        - The method requires both observed and expected frequency tables.
        - The returned value is rounded to 3 decimal places for readability.

        Example
        -------
        observed_crosstab = obj.crosstab('variable1', 'variable2')
        expected_crosstab = MyClass.expected_crosstab(observed_crosstab)
        adjusted_res = MyClass.adjusted_residual(observed_crosstab, expected_crosstab, i=2, j=3)
        # Use `adjusted_res` to assess the significance of the difference between the observed and expected values.
        """
        row_totals = [item for item in observed_crosstab.sum(axis=1)]
        col_totals = [item for item in observed_crosstab.sum(axis=0)]
        n = sum(row_totals)

        adjusted_residual = (
                (observed_crosstab.iloc[i, j] - expected_crosstab.iloc[i, j])
                / (
                        expected_crosstab.iloc[i, j]
                        * (1 - row_totals[i] / n)
                        * (1 - col_totals[j] / n)
                )
                ** 0.5
        )

        return round(adjusted_residual, 3)

    def chi2_post_hoc_test(self, p_val, crosstab_input, expected_crosstab):
        """
        Perform a Chi-squared post hoc test to identify significant differences between categories of a variable.

        This method helps determine which categories within a significant variable (question) have significant differences
        based on the p-value obtained from a Chi-squared contingency test.

        Parameters
        ----------
        p_val : float
            The p-value obtained from a Chi-squared contingency test (e.g., scipy.stats.chi2_contingency).
        crosstab_input : pandas.DataFrame
            Contingency table with observed frequencies of two variables.
        expected_crosstab : pandas.DataFrame
            Contingency table with expected frequencies of two variables.

        Returns
        -------
        pandas.DataFrame or None
            Contingency table with test results in each cell, indicating if categories are:
            - 'pos' (significantly above expected value)
            - 'neg' (significantly below expected value)
            - 'neu' (within the expected range)

            Returns None if the p-value is not significant at the specified significance level.

        Notes
        -----
        - This method performs a Chi-squared post hoc test to assess the significance of differences between categories.
        - It considers both the adjusted residuals and p-values for each cell in the contingency table.
        - Significance is determined based on the p-value and the chosen correction method (e.g., Bonferroni).
        - If the p-value is not significant, the method returns None.
        - The `correction` attribute of the class controls the correction method used (e.g., 'bonferroni').

        Example
        -------
        observed_crosstab = obj.crosstab('variable1', 'variable2')
        expected_crosstab = MyClass.expected_crosstab(observed_crosstab)
        p_value = MyClass.chi2_stats(observed_crosstab, expected_crosstab)[1]
        post_hoc_results = obj.chi2_post_hoc_test(p_value, observed_crosstab, expected_crosstab)
        if post_hoc_results is not None:
            # Analyze and interpret the post hoc test results.
        else:
            # No significant differences found; continue analysis accordingly.
        """

        alpha = 1.0 - self.ci

        if p_val > alpha:
            return None  # Return early if not significant

        n_rows, n_cols = crosstab_input.shape

        residuals_list = []
        p_value_list = []

        direction = self.direction(
            crosstab_input=crosstab_input.copy(deep=True),
            expected_crosstab=expected_crosstab.copy(deep=True),
        )

        for i in range(n_rows):
            residuals = []
            p_values = []

            for j in range(n_cols):
                adj_res = self.adjusted_residual(crosstab_input, expected_crosstab, i, j)
                ind_chi_square = adj_res * adj_res
                p_ind_chi_square = 1 - stats.chi2.cdf(ind_chi_square, 1)
                z_score_abs = abs(adj_res)

                if self.correction == 'bonferroni':
                    adjusted_p = alpha / (n_rows * n_cols)
                    p_values.append(adjusted_p)

                    if (z_score_abs >= 1.96) and (p_ind_chi_square <= adjusted_p):
                        residuals.append("pos" if direction.iloc[i, j] else "neg")
                    else:
                        residuals.append("neu")
                else:
                    if z_score_abs >= 1.96:
                        residuals.append("pos" if direction.iloc[i, j] else "neg")
                    else:
                        residuals.append("neu")

            residuals_list.append(residuals)
            p_value_list.append(p_values)

        residuals_df = pd.DataFrame(residuals_list, index=crosstab_input.index, columns=crosstab_input.columns)
        return residuals_df

    def json_cluster_results(self):
        """
        Execute the Chi-squared contingency table statistic and perform post hoc tests for clusters.
        Store the results in a JSON-type dictionary format within the class.

        Returns
        -------
        dict
            A JSON-style dictionary containing cluster-specific information in the following format:

            {
                'cluster 1': {
                    'proportion': 558,
                    'weighted_proportion': None,
                    'percentage_proportion': 37.78,
                    'weighted_percentage_proportion': None,
                    'variables': {
                        '4': {
                            'chi2_stat': 0.0,
                            'p_val': 1.0,
                            'var_significance': False,
                            'TGT': False,
                            'pop_mode': '10003',
                            'pop_mode_perc': 3.58,
                            'weighted_pop_mode': None,
                            'weighted_pop_mode_perc': None,
                            'categories': {},
                            'target': False
                        },
                        '6': {
                            'chi2_stat': 168.7,
                            'p_val': 0.0,
                            'var_significance': True,
                            'TGT': False,
                            'pop_mode': '10008',
                            'pop_mode_perc': 39.25,
                            'weighted_pop_mode': None,
                            'weighted_pop_mode_perc': None,
                            'categories': {
                                '10006': {
                                    'significance': True,
                                    'frequency_count': 51,
                                    'expected_freq_count': 85.54,
                                    'frequency_percentage': 9.5,
                                    'post_hoc_test': 'neg'
                                },
                                '10007': {
                                    'significance': False,
                                    'frequency_count': 178,
                                    'expected_freq_count': 194.83,
                                    'frequency_percentage': 33.1,
                                    'post_hoc_test': 'neu'
                                },
                                '10008': {
                                    'significance': True,
                                    'frequency_count': 219,
                                    'expected_freq_count': 177.87,
                                    'frequency_percentage': 40.8,
                                    'post_hoc_test': 'pos'
                                },
                                '10009': {
                                    'significance': False,
                                    'frequency_count': 89,
                                    'expected_freq_count': 78.76,
                                    'frequency_percentage': 16.6,
                                    'post_hoc_test': 'neu'
                                }
                            },
                            'target': True
                        }
                    }
                },
                ...
            }

        Notes
        -----
        - This method performs a Chi-squared contingency table statistic and post hoc tests for clusters.
        - The results are stored in a dictionary format within the class.
        - The dictionary contains cluster-specific information, including proportion, Chi-squared statistics, p-values,
        variable significance, and more.
        - The method utilizes parallel processing to improve efficiency when analyzing multiple clusters.

        Example
        -------
        json_results = obj.json_cluster_results()
        if json_results:
            # Access and analyze cluster-specific results from the dictionary.
        else:
            # No clusters detected; handle accordingly.
        """
        variables = self.data.columns.tolist()
        try:
            variables.remove(self.cluster_col)
        except (KeyError, ValueError) as e:
            log.info(
                "json_cluster_results - 2. Could not remove cluster_col from variables - column doesn't exist?"
            )

        to_exclude = self.inference_excluded_cols()
        variables = list(set(variables) - set(to_exclude))

        results = {}

        if not self.data[self.cluster_col].empty:
            cluster_labels = np.sort(
                self.data[self.cluster_col].unique().tolist()
            )  # todo: this throws error in python=3.10 ????

            result_list = Parallel(n_jobs=-1)(
                delayed(self.json_cluster_results_per_segment)(c, variables)
                for c in cluster_labels
            )

            # result_list = []  # todo: use for LOCAL DEBUG
            # for c in cluster_labels:
            #     result = self.json_cluster_results_per_segment(cluster=c, variables=variables)
            #     result_list.append(result)

            for i in range(len(result_list)):
                for k, v in result_list[i].items():
                    results[k] = v
        else:
            log.info("json_cluster_results - could not detect clusters")

        self.cluster_results = results

        return results

    def flag_not_selected(self, cluster, q_ref_table, req_shortname):
        if 'act_time_' in req_shortname:
            return
        if req_shortname not in self.data.columns:
            print(f'{req_shortname} not in the data - check questions file for missing data')
            return
        full_data = self.data_labelled.copy()
        full_data = full_data[full_data[self.segmentation] == cluster]
        full_data.columns = full_data.columns.str.replace('_tgt ', '')

        full_data.columns = [col.replace('_tgt', '') for col in full_data.columns]
        # mask = (full_data!= "Not shown")

        required_varnames = list(q_ref_table[q_ref_table['shortname'] == req_shortname]['varname'].unique())

        if not required_varnames:
            raise ValueError(f'No data for the question: {req_shortname}')

        set_required_varnames = set(required_varnames)
        set_full_data_columns = set(full_data.columns)

        required_varnames_match = set_required_varnames.intersection(set_full_data_columns)
        excluded_varnames = set_required_varnames - required_varnames_match

        required_varnames_match_list = list(required_varnames_match)

        if not required_varnames_match_list:
            required_varnames_match_list = [col for col in set_full_data_columns if req_shortname in col]

        if excluded_varnames:
            print(
                f'WARNING: {excluded_varnames} missing from responses data but is in question bank, may need to inspect.')

        if self.weights:
            required_cols = [self.weights] + required_varnames_match_list
        else:
            required_cols = required_varnames_match_list

        if len(required_cols) == 0:
            return

        req_data = full_data[required_cols].copy()

        try:
            req_data['row_values'] = req_data[required_varnames_match_list].stack().groupby(level=0).apply(
                lambda x: x.unique().tolist())
        except:
            print('inspect')

        try:
            req_data['row_values_clean'] = req_data['row_values'].apply(
                lambda x: x if x == ['not selected'] else [i for i in x if i != 'not selected'])
        except:
            print('inspect')

        req_data['row_values_clean'] = req_data['row_values'].apply(
            lambda x: x if x == ['not selected'] else [i for i in x if i != 'not selected'])
        if 'precompletion_weight' not in req_data.columns:
            req_data['precompletion_weight'] = 1
        req_data_explode = req_data[['precompletion_weight', 'row_values_clean']].explode('row_values_clean')

        try:
            weighted_sample_total = round(self.data[self.weights].sum(), 2)
        except KeyError:
            weighted_sample_total = len(self.data)

        weighted_counts = req_data_explode.groupby(['row_values_clean'])[
                              'precompletion_weight'].sum() / weighted_sample_total
        weighted_mode = weighted_counts.idxmax()
        weighted_mode_prop = round(weighted_counts[weighted_mode], 2)

        sample_total = self.data.shape[0]
        unweighted_counts = req_data_explode.groupby(['row_values_clean'])['row_values_clean'].count() / sample_total
        unweighted_mode = unweighted_counts.idxmax()
        unweighted_mode_prop = round(unweighted_counts[unweighted_mode], 2)

        final_dict = {}
        final_dict['shortname'] = req_shortname
        final_dict['weighted_mode'] = weighted_mode
        final_dict['weighted_mode_prop'] = weighted_mode_prop
        final_dict['unweighted_mode'] = unweighted_mode
        final_dict['unweighted_mode_prop'] = unweighted_mode_prop

        # haven't considered 'Not shown' responses

        return final_dict

    def check_pop_modes_exist(self):
        if s3_check.exists(
                f's3://qudo-datascience/data-store/pythia_exploration/{self.env}/{self.survey_name}/population_modes/population_modes.parquet'):
            return True
        return False

    def json_sample_results(self):

        sp_tag = False
        if "_sp_" in str(self.survey_name).lower():
            sp_tag = True

        if sp_tag:
            trunc_surv_name = "_".join(self.survey_name.split("_")[0:-2])
            question_ref = pd.read_parquet(
                f's3://qudo-datascience/data-store/special_projects/{self.env}/{self.survey_name.split("_")[0]}/{trunc_surv_name}/{trunc_surv_name}_questions_preprocessed.parquet')
        else:
            question_ref = pd.read_parquet(
                f's3://qudo-datascience/data-store/surveys/{self.env}/questions_preprocessed/{self.survey_name}/{self.survey_name}.parquet')

        undesired_q_categories = ['att', 'qudo', 'ref']
        varname_shortname_mapping = question_ref[~question_ref['category'].isin(undesired_q_categories)][
            ['varname', 'shortname']].copy().drop_duplicates()

        shortnames = list(set(varname_shortname_mapping['shortname']))

        clusters = self.data_labelled[self.segmentation].unique()

        final_results = {}
        for cluster in clusters:
            result_list = Parallel(n_jobs=-1)(
                delayed(self.flag_not_selected)(cluster=cluster, q_ref_table=varname_shortname_mapping,
                                                req_shortname=req_shortname)
                for req_shortname in shortnames
            )

            result_list = list(filter(lambda item: item is not None, result_list))
            results_df = pd.DataFrame(result_list)
            results_df['cluster'] = cluster
            results_df['survey_name'] = self.survey_name
            results_df['segmentation'] = self.segmentation
            final_results[cluster] = results_df

        final_results_df = pd.concat(final_results.values())
        final_results_df[['weighted_mode', 'unweighted_mode']] = final_results_df[
            ['weighted_mode', 'unweighted_mode']].applymap(strip_html)

        # result_list = Parallel(n_jobs=-1)(
        #     delayed(self.flag_not_selected)(varname_shortname_mapping, req_shortname)
        #     for req_shortname in shortnames
        # )
        #
        # results_df = pd.DataFrame(result_list)
        # results_df['survey_name'] = self.survey_name

        final_results_df.to_parquet(
            f's3://qudo-datascience/data-store/pythia_exploration/{self.env}/{self.survey_name}/population_modes/population_modes.parquet')

    def calculate_weighted_cluster_proportion(self, cluster):
        """
        Calculate the weighted proportion and weighted percentage proportion of a specific cluster.

        Parameters
        ----------
        cluster : str or int
            The cluster identifier for which to calculate proportions.

        Returns
        -------
        tuple
            A tuple containing two values:
            - weighted_prop (float): The weighted proportion of the specified cluster.
            - weighted_percentage_prop (float): The weighted percentage proportion of the specified cluster.

        Notes
        -----
        - This method calculates the weighted proportion and weighted percentage proportion for a specific cluster.
        - The weighted proportion is the sum of weights for records within the cluster.
        - The weighted percentage proportion represents the weighted proportion as a percentage of the total sample.

        Example
        -------
        cluster = 'cluster1'
        weighted_prop, weighted_percentage_prop = obj.calculate_weighted_cluster_proportion(cluster)
        # Use `weighted_prop` and `weighted_percentage_prop` for further analysis.
        """

        sample_total = self.data.shape[0]

        weighted_prop = round(self.data[self.data[self.cluster_col] == cluster][self.weights].sum(), 2)

        weighted_percentage_prop = round(
            (weighted_prop / sample_total) * 100, 2
        )

        return weighted_prop, weighted_percentage_prop

    def calculate_variable_mode(self, cluster, v, proportion):
        """
        Calculate the mode and its percentage proportion for a variable within a specific cluster.

        Parameters
        ----------
        cluster : str or int
            The cluster identifier for which to calculate the variable mode.
        v : str
            The name of the variable for which to calculate the mode.
        proportion : float
            The total proportion of records within the cluster.

        Returns
        -------
        tuple
            A tuple containing two values:
            - pop_mode (str): The mode value of the specified variable within the cluster.
            - pop_mode_perc (float): The percentage proportion of the mode value within the cluster.

        Notes
        -----
        - This method calculates the mode value and its percentage proportion for a specific variable within a cluster.
        - The mode is the most frequently occurring value within the cluster, excluding "Not shown" values.
        - The percentage proportion represents the mode's occurrence as a percentage of the total records in the cluster.

        Example
        -------
        cluster = 'cluster1'
        variable = 'variable1'
        total_proportion = 210  # Total proportion of records in 'cluster1'
        pop_mode, pop_mode_perc = obj.calculate_variable_mode(cluster, variable, total_proportion)
        # Use `pop_mode` and `pop_mode_perc` for further analysis.
        """

        mask = (self.data[self.cluster_col] == cluster) & (
                self.data[v] != "Not shown"
        )
        pop_mode = self.data[mask][v].mode().values.tolist()[0]
        pop_mode_perc = round(
            (self.data[mask][v].value_counts()[pop_mode] / proportion)
            * 100,
            2,
        )

        return pop_mode, pop_mode_perc

    def calculate_weighted_variable_mode(self, cluster, v, proportion):
        """
        Calculate the weighted mode and its percentage proportion for a variable within a specific cluster.

        Parameters
        ----------
        cluster : str or int
            The cluster identifier for which to calculate the weighted variable mode.
        v : str
            The name of the variable for which to calculate the weighted mode.
        proportion : float
            The total proportion of records within the cluster.

        Returns
        -------
        tuple
            A tuple containing two values:
            - weighted_pop_mode (str): The weighted mode value of the specified variable within the cluster.
            - weighted_pop_mode_perc (float): The percentage proportion of the weighted mode value within the cluster.

        Notes
        -----
        - This method calculates the weighted mode value and its percentage proportion for a specific variable within a cluster.
        - The weighted mode is determined by finding the category with the highest weighted sum, excluding "Not shown" values.
        - The percentage proportion represents the weighted mode's occurrence as a percentage of the total records in the cluster.

        Example
        -------
        cluster = 'cluster1'
        variable = 'variable1'
        total_proportion = 200  # Total proportion of records in 'cluster1'
        weighted_pop_mode, weighted_pop_mode_perc = obj.calculate_weighted_variable_mode(cluster, variable, total_proportion)
        # Use `weighted_pop_mode` and `weighted_pop_mode_perc` for further analysis.
        """
        mask = (self.data[self.cluster_col] == cluster) & (
                self.data[v] != "Not shown"
        )

        category_sizes = self.data[mask].groupby(v)[[self.weights]].sum() / proportion * 100
        category_sizes = category_sizes.reset_index()
        category_sizes.columns = ["category", "value"]

        weighted_pop_mode = category_sizes.max()["category"]
        weighted_pop_mode_perc = round(category_sizes.max()["value"])

        return weighted_pop_mode, weighted_pop_mode_perc

    def append_post_hoc_results(self, p, crosstab, cluster, cat, var):
        """
        Append post hoc test results for a specific category within a cluster to a dictionary.

        Parameters
        ----------
        p : float
            The p-value of the Chi-squared contingency test for the inputted crosstab.
        crosstab : pd.DataFrame
            A crosstab of observed frequencies.
        cluster : str or int
            The cluster identifier.
        cat : str
            The category within the variable.
        var : str
            The variable within the clustered data.

        Returns
        -------
        dict
            A dictionary containing post hoc test results for the specified category within the cluster.

        Notes
        -----
        - This method calculates and appends post hoc test results to a dictionary.
        - Post hoc test results include significance, frequency counts, expected frequency counts, frequency percentages,
              and the post hoc test outcome.
        - If weights are provided, weighted frequency counts and percentages are also included.

        Example
        -------
        p_value = 0.05
        crosstab_table = pd.crosstab(...)
        cluster_id = 'cluster1'
        category = 'categoryA'
        variable_name = 'variableX'
        post_hoc_results = obj.append_post_hoc_results(p_value, crosstab_table, cluster_id, category, variable_name)
        # Access post hoc test results from the dictionary for further analysis.
        """
        expected_crosstab = self.expected_crosstab(
            crosstab_input=crosstab.copy(deep=True)
        )
        percent_crosstab = self.crosstab_percent(
            crosstab_input=crosstab.copy(deep=True)
        )

        post_hoc = self.chi2_post_hoc_test(
            p_val=p,
            crosstab_input=crosstab.copy(deep=True),
            expected_crosstab=expected_crosstab.copy(deep=True),
        )

        if post_hoc.loc[cat, cluster] == "neu":
            sig = False
        else:
            sig = True

        subset = self.data[self.data[self.cluster_col] == cluster].copy(deep=True)
        try:
            true_seg_frequency_percentage = dict(subset[var].value_counts(normalize=True) * 100)[cat]
        except KeyError:
            true_seg_frequency_percentage = 0.0

        crosstab_frequency_percentage = round(
            percent_crosstab.loc[cat, cluster], 2
        )

        post_hoc_dict = {
            "significance": sig,
            "frequency_count": int(crosstab.loc[cat, cluster]),
            "expected_freq_count": round(
                expected_crosstab.loc[cat, cluster], 2
            ),
            "frequency_percentage": round(true_seg_frequency_percentage, 2),
            "post_hoc_test": post_hoc.loc[cat, cluster],
        }

        if self.weights:
            weighted_frequency_count = subset[subset[var] == cat][self.weights].sum()
            weighted_seg_frequency_percentage = weighted_frequency_count / subset[self.weights].sum() * 100

            post_hoc_dict["weighted_frequency_count"] = weighted_frequency_count
            post_hoc_dict["weighted_frequency_percentage"] = round(weighted_seg_frequency_percentage, 2)

        return post_hoc_dict

    @staticmethod
    def append_target_value(results, cluster, v, categories):
        """
        Append a target value to cluster results based on post hoc test outcomes.

        Parameters
        ----------
        results : dict
            A dictionary containing cluster-specific information.
        cluster : str or int
            The cluster identifier.
        v : str
            The variable within the cluster.
        categories : list of str
            A list of categories within the variable.

        Returns
        -------
        dict
            The updated dictionary with a "target" value added to cluster results for the specified variable.

        Notes
        -----
        - This method examines post hoc test outcomes for categories within a variable and determines a "target" value.
        - If any category has a "pos" post hoc test outcome (significantly above expected counts), the "target" value is set to True.
        - If all categories have a "neu" post hoc test outcome (within expected counts), the "target" value is set to False.

        Example
        -------
        cluster_results = {...}  # Dictionary containing cluster-specific information.
        cluster_id = 'cluster1'
        variable_name = 'variableX'
        categories_list = ['categoryA', 'categoryB', 'categoryC']
        updated_results = MyClass.append_target_value(cluster_results, cluster_id, variable_name, categories_list)
        # Access the updated cluster results with the "target" value.
        """
        post_hoc_results = [
            results[cluster]["variables"][v]["categories"][cat][
                "post_hoc_test"
            ]
            for cat in categories
        ]
        if "pos" in post_hoc_results:  # or ('neg' in post_hoc_results):
            # todo: at the moment this only considers more than expected counts
            results[cluster]["variables"][v]["target"] = True
        else:
            results[cluster]["variables"][v]["target"] = False

        return results

    def json_cluster_results_per_segment(self, cluster, variables):
        """
        Calculate and collect cluster-specific information for a given cluster label and variables.

        Parameters
        ----------
        cluster : str or int
            The cluster label for a segment.
        variables : list
            A list of variables to analyze within the cluster.

        Returns
        -------
        dict
            A dictionary containing cluster-specific information, including variable statistics and post hoc test results.

        Notes
        -----
        - This method calculates and collects information for a specific cluster, including variable statistics,
        significance tests, mode values, and post hoc test results.
        - The results are organized in a dictionary format with cluster-specific details.

        Example
        -------
        cluster_id = 'cluster1'
        variable_list = ['var1', 'var2', 'var3']
        cluster_info = obj.json_cluster_results_per_segment(cluster_id, variable_list)
        # Access cluster-specific information from the dictionary.
        """
        alpha = 1.0 - self.ci
        sample_total = self.data.shape[0]
        proportion = len(self.data[self.data[self.cluster_col] == cluster])
        weight_col = self.weights
        data = self.data.copy()
        cluster_col = self.cluster_col

        results = {
            cluster: {"proportion": proportion, "percentage_proportion": round((proportion / sample_total) * 100, 2),
                      "variables": {}}}

        if weight_col:
            weighted_prop, weighted_percentage_proportion = self.calculate_weighted_cluster_proportion(cluster=cluster)
            results[cluster]["weighted_proportion"] = weighted_prop
            results[cluster]["weighted_percentage_proportion"] = weighted_percentage_proportion

        variable_count = 0
        tic = time.perf_counter()
        for v in variables:
            variable_count += 1

            crosstab, Yates_correction = self.crosstab(var=v)

            if Yates_correction:
                stat, p, _, _ = self.chi2_stats(crosstab_input=crosstab, yates=True)
            else:
                stat, p, _, _ = self.chi2_stats(crosstab_input=crosstab)

            if p > alpha:
                del p
                continue

            if stat is not None and cluster in crosstab.columns:
                pop_mode, pop_mode_perc = self.calculate_variable_mode(cluster=cluster, v=v, proportion=proportion)
                results[cluster]["variables"][v] = {"chi2_stat": round(stat, 2), "p_val": round(p, 5),
                                                    "var_significance": p <= alpha, "pop_mode": pop_mode,
                                                    "pop_mode_perc": pop_mode_perc, "categories": {}}

                if weight_col:
                    weighted_proportion = data[data[cluster_col] == cluster][weight_col].sum()
                    weighted_pop_mode, weighted_pop_mode_perc = self.calculate_weighted_variable_mode(cluster=cluster,
                                                                                                      v=v,
                                                                                                      proportion=weighted_proportion)
                    results[cluster]["variables"][v]["weighted_pop_mode"] = weighted_pop_mode
                    results[cluster]["variables"][v]["weighted_pop_mode_perc"] = weighted_pop_mode_perc

                if p <= alpha:
                    categories = crosstab.index.tolist()
                    results[cluster]["variables"][v]["categories"] = {
                        cat: self.append_post_hoc_results(p=p, crosstab=crosstab, cluster=cluster, cat=cat, var=v) for
                        cat in categories}
                    results = self.append_target_value(results=results, cluster=cluster, v=v, categories=categories)

                    del p

        toc = time.perf_counter()
        log.info(
            f"json_cluster_results_per_segment - the chi2 stats calculation took {toc - tic:0.4f} seconds for {variable_count} variables")
        if not results:
            log.info("json_cluster_results_per_segment - results empty.")
        return results

    def build_deliver_stats_dict(self, deliver_stats_dict, q_code, segment_id, variable_results, return_chi2_stat):
        """
        Build a dictionary of deliverable statistics for a specific question code, segment, and variable results.

        Parameters
        ----------
        deliver_stats_dict : dict
            A dictionary containing deliverable statistics.
        q_code : str
            The question code identifier.
        segment_id : str or int
            The segment identifier.
        variable_results : dict
            A dictionary containing variable-specific results.
        return_chi2_stat : bool
            Whether to include the Chi-squared statistic in the dictionary.

        Returns
        -------
        dict
            An updated dictionary containing deliverable statistics for the specified question code, segment, and variable.

        Notes
        -----
        - This method compiles and updates a dictionary with deliverable statistics for a given question code,
            segment identifier, and variable results.
        - The resulting dictionary includes information such as response rate, mode population percentage,
            Chi-squared result, significant categories, and targeting segments.

        Example
        -------
        deliverable_stats = {...}  # Dictionary containing deliverable statistics.
        question_code = 'Q123'
        segment_identifier = 'segment1'
        variable_results_data = {...}  # Variable-specific results in a dictionary.
        include_chi2_stat = True
        updated_stats = obj.build_deliver_stats_dict(deliverable_stats, question_code, segment_identifier,
                                                        variable_results_data, include_chi2_stat)
        # Access the updated dictionary of deliverable statistics.
        """
        response_rate = self.data[q_code].count() / len(self.data[q_code]) * 100
        pos_cats = []
        pos_percentages = []
        weighted_pos_percentages = []

        for cat, cat_data in variable_results["categories"].items():
            if cat_data["post_hoc_test"] == "pos":
                pos_cats.append(cat)
                pos_percentages.append(cat_data["frequency_percentage"])
                if self.weights:
                    weighted_pos_percentages.append(cat_data["weighted_frequency_percentage"])

        deliver_stats_dict["q_code"].append(q_code)
        deliver_stats_dict["pop_mode"].append(variable_results["pop_mode"])
        deliver_stats_dict["response_rate"].append(response_rate)
        deliver_stats_dict["mode_pop_perc"].append(variable_results["pop_mode_perc"])
        deliver_stats_dict["chi_2_result"].append(variable_results["p_val"])
        deliver_stats_dict["sig_more_category"].append(pos_cats)
        deliver_stats_dict["category_percentages"].append(pos_percentages)
        deliver_stats_dict["targeting_seg"].append(segment_id)

        if return_chi2_stat:
            deliver_stats_dict['chi2_stat'].append(variable_results['chi2_stat'])

        if self.weights:
            deliver_stats_dict["weighted_pop_mode"].append(variable_results["weighted_pop_mode"])
            deliver_stats_dict["weighted_pop_mode_perc"].append(variable_results["weighted_pop_mode_perc"])
            deliver_stats_dict["weighted_category_percentages"].append(weighted_pos_percentages)

        return deliver_stats_dict

    def extract_deliver_stats_df(self, return_chi_2_stat=False) -> bool:
        """
        This function returns a dataframe with all the chi2 significant variables and other relevant information needed
        for the frontend and further modules such as segmentation metrics, Sophos description generation or Infogrpahic
        text.

        Returns
        -------
        bool
            True if the dataframe with significant variables is generated, False otherwise.

        self.deliver_pg_stats: pandas.DataFrame
            Indirectly returned as a class object, derived from summary_stats_df.


        Notes
        ----
            The returned dataframe contains the following variables:
                q_code:
                    This is the question code that has been given when ingested into the class.
                    Data type: string of data code
                pop_mode:
                    This is the mode answer of the question of q_code within the targeting segment (targeting_seg),
                    NOT within the population.
                    N.B.: This has been named pop_mode because previously this was the mode of the whole dataset NOT
                    within the segment. This may be refactored to "seg_mode".
                    Data type: String of answer code
                    # todo: refactor variable name to seg_mode. Check for frontend compatibility of name and value.
                response_rate:
                    This is the response rate of a question (q_code) within the whole dataset NOT the targeting segment
                    itself. This may be amended to response rate of the segment itself to a certain question (q_code).
                    # todo: change to response rate within segment. Has to be checked for frontend compatibility.
                mode_pop_perc:
                    This is the percentage of the specified targeting segment (targeting_seg) that has the mode value
                    within the variable (q_code).
                    N.B.: This has been named mode_pop_perc because previously this was its name. This was also
                    previously the percentage of the whole dataset containing the mode value for the specified variable
                    (q_code) and NOT just within the targeting segment. This may be refactored to "mode_seg_perc".
                    Data type: Float
                    # todo: refactor variable name to mode_seg_perc. Check for frontend compatibility of name and value.
                chi2_2_result:
                    This is the p-value of a question (q_code) determined from the chi2 contingency table statistic.
                    The lower the p-value the more significant the question answers differed between the segments.
                sig_more_category:
                    These are the values (i.e. categories) within the variable (q_code) that have been significantly
                    "more" different to their expected values and influenced the variable (q_code) to be determined as
                    significant.
                    Data type: List of answers, usually they are codes (strings) depending on what has been ingested.
                targeting_seg:
                    This is the segment (cluster) that the categories (sig_more_category) for each question (q_code)
                    will be targeted on.
                    Data type: String of segment, usually they are codes (strings) depending on what has been ingested.

        """

        deliver_stats_dict = {
            "q_code": [],
            "pop_mode": [],
            "response_rate": [],
            "mode_pop_perc": [],
            "chi_2_result": [],
            "sig_more_category": [],
            "category_percentages": [],
            "targeting_seg": [],
        }

        if return_chi_2_stat:
            deliver_stats_dict['chi2_stat'] = []

        if self.weights:
            deliver_stats_dict["weighted_pop_mode"] = []
            deliver_stats_dict["weighted_pop_mode_perc"] = []
            deliver_stats_dict["weighted_category_percentages"] = []

        tic = time.perf_counter()
        cluster_results = self.json_cluster_results()
        toc = time.perf_counter()
        log.info(f"json_cluster_results() finished in {toc - tic:0.4f} seconds")

        for segment_id, segment_data in cluster_results.items():
            for q_code, variable_results in segment_data["variables"].items():
                deliver_stats_dict = self.build_deliver_stats_dict(deliver_stats_dict=deliver_stats_dict,
                                                                       q_code=q_code,
                                                                       segment_id=segment_id,
                                                                       variable_results=variable_results,
                                                                       return_chi2_stat=return_chi_2_stat)

            # deliver_stats_dict = Parallel(n_jobs=-1)(  # TODO: errors parallelized sometimes :(
            #         delayed(self.build_deliver_stats_dict)(deliver_stats_dict=deliver_stats_dict,
            #                                                q_code=q_code,
            #                                                segment_id=segment_id,
            #                                                variable_results=variable_results,
            #                                                return_chi2_stat=return_chi_2_stat)
            #         for q_code, variable_results in segment_data["variables"].items()
            # )

        if not deliver_stats_dict["targeting_seg"]:
            return False

        summary_stats_df = pd.DataFrame(deliver_stats_dict).sort_values("chi_2_result")

        summary_stats_df = summary_stats_df.apply(remove_not_select_and_cat_percentage, axis=1)

        # assessing if any significance has been reported
        if summary_stats_df.shape[0] == 0:
            pass
        else:
            """removing any empty list entries within sig_more_category from dataframe"""
            summary_stats_df = summary_stats_df[
                summary_stats_df["sig_more_category"].str.len() != 0
                ]

        if summary_stats_df.empty:
            log.info("extract_deliver_stats_df - summary_stats_df empty.")
            return False

        self.deliver_pg_stats = summary_stats_df
        return True

    def inference_schema_df(self, return_chi_2_stat):
        """
        Generate an empty schema dataframe for inference.

        This method creates an empty schema dataframe with predefined columns for inference results.

        Parameters
        ----------
        return_chi_2_stat : bool
            Whether to include the Chi-squared statistic column in the schema dataframe.

        Returns
        -------
        schema_df : pd.DataFrame
            An empty schema dataframe with predefined columns for inference results.

        Example
        -------
        schema = obj.inference_schema_df(return_chi_2_stat=True)
        # Use the schema to structure and store inference results.
        """
        schema_df = pd.DataFrame({
            "q_code": [],
            "pop_mode": [],
            "response_rate": [],
            "mode_pop_perc": [],
            "chi_2_result": [],
            "sig_more_category": [],
            "category_percentages": [],
            "targeting_seg": []
        })

        if return_chi_2_stat:
            schema_df['chi2_stat'] = []

        if self.weights:
            schema_df["weighted_pop_mode"] = []
            schema_df["weighted_pop_mode_perc"] = []
            schema_df["weighted_category_percentages"] = []

        self.schema_df = schema_df

        return schema_df

    def generate_lazy_segment_inference(self, cluster, variables):
        """This is the lazy method of calculating the inference_data DataFrame per cluster.
        It only executes calculations if:
            p <= alpha
            post_hoc result == 'pos'

        Parameters
        ----------
        cluster : str, int
            The cluster label for which inference data will be generated.
        variables : list
            A list of significant variables to consider for inference.

        Returns
        -------
        segment_results_df : pd.DataFrame
            A DataFrame containing inference data for the specified cluster and significant variables.

        Example
        -------
        inference_data = obj.generate_lazy_segment_inference(cluster="cluster1", variables=["var1", "var2"])
        # Use the inference_data for analysis and visualization.
        """
        alpha = 1.0 - self.ci
        data = self.data.copy()
        cluster_col = self.cluster_col
        weight_col = self.weights

        sample_total = data.shape[0]
        subset = data[data[cluster_col] == cluster].copy(deep=True)
        proportion = subset.shape[0]
        percentage_proportion = round((proportion / sample_total) * 100, 2)

        segment_results_df = self.schema_df.copy()
        q_codes = []
        pop_modes = []
        response_rates = []
        mode_pop_percs = []
        chi_2_results = []
        sig_more_categories = []
        category_percentages = []
        targeting_segs = []

        # conditionally added
        chi2_stats = []
        weighted_pop_modes = []
        weighted_pop_mode_percs = []
        weighted_category_percentages = []


        variable_count = 0
        tic = time.perf_counter()
        for v in variables:
            variable_count += 1

            crosstab, Yates_correction = self.crosstab(var=v)

            if Yates_correction:
                stat, p, _, _ = self.chi2_stats(crosstab_input=crosstab, yates=True)
            else:
                stat, p, _, _ = self.chi2_stats(crosstab_input=crosstab)

            if p > alpha:
                continue

            if stat is not None and cluster in crosstab.columns:
                # calc mode and perc
                pop_mode, pop_mode_perc = self.calculate_variable_mode(cluster=cluster, v=v, proportion=proportion)
                # overall response rate of variable NOT within segment... # todo: discuss this and pop_mode
                response_rate = self.data[v].count() / len(self.data[v]) * 100

                # extract post hoc results
                expected_crosstab = self.expected_crosstab(
                    crosstab_input=crosstab.copy(deep=True)
                )

                post_hoc = self.chi2_post_hoc_test(
                    p_val=p,
                    crosstab_input=crosstab.copy(deep=True),
                    expected_crosstab=expected_crosstab.copy(deep=True),
                )

                categories = crosstab.index.tolist()
                pos_cats = []
                category_percs = []
                weighted_category_percs = []
                for cat in categories:
                    if post_hoc.loc[cat, cluster] == "pos":

                        pos_cats.append(cat)

                        try:
                            true_seg_frequency_percentage = dict(subset[v].value_counts(normalize=True) * 100)[cat]
                            true_seg_frequency_percentage = round(true_seg_frequency_percentage, 2)
                        except KeyError:
                            true_seg_frequency_percentage = 0.0

                        category_percs.append(true_seg_frequency_percentage)

                        if weight_col:
                            weighted_frequency_count = subset[subset[v] == cat][weight_col].sum()
                            weighted_seg_frequency_percentage = round(weighted_frequency_count / subset[
                                weight_col].sum() * 100, 2)

                            weighted_category_percs.append(weighted_seg_frequency_percentage)

                    else:
                        pass

                q_codes.append(v)
                pop_modes.append(pop_mode)
                response_rates.append(response_rate)
                mode_pop_percs.append(pop_mode_perc)
                chi_2_results.append(round(p, 5))
                sig_more_categories.append(pos_cats)
                category_percentages.append(category_percs)
                targeting_segs.append(cluster)

                if "chi2_stat" in segment_results_df.columns:
                    chi2_stats.append(round(stat, 2))

                if weight_col:
                    weighted_proportion = data[data[cluster_col] == cluster][weight_col].sum()
                    weighted_pop_mode, weighted_pop_mode_perc = self.calculate_weighted_variable_mode(
                            cluster=cluster,
                            v=v,
                            proportion=weighted_proportion)

                    weighted_pop_modes.append(weighted_pop_mode)
                    weighted_pop_mode_percs.append(weighted_pop_mode_perc)
                    weighted_category_percentages.append(weighted_category_percs)

        toc = time.perf_counter()
        log.info(
            f"generate_lazy_segment_inference - the chi2 stats calculation took {toc - tic:0.4f} seconds for {variable_count} variables")

        segment_results_df["q_code"] = q_codes
        segment_results_df["pop_mode"] = pop_modes
        segment_results_df["response_rate"] = response_rates
        segment_results_df["mode_pop_perc"] = mode_pop_percs
        segment_results_df["chi_2_result"] = chi_2_results
        segment_results_df["sig_more_category"] = sig_more_categories
        segment_results_df["category_percentages"] = category_percentages
        segment_results_df["targeting_seg"] = targeting_segs

        if "chi2_stats" in segment_results_df.columns:
            segment_results_df['chi2_stat'] = chi2_stats

        if weight_col:
            segment_results_df["weighted_pop_mode"] = weighted_pop_modes
            segment_results_df["weighted_pop_mode_perc"] = weighted_pop_mode_percs
            segment_results_df["weighted_category_percentages"] = weighted_category_percentages

        if segment_results_df.empty:
            log.info("generate_lazy_segment_inference - segment_results_df empty.")
        return segment_results_df

    def lazy_calculate_inference_data(self, return_chi_2_stat=False) -> bool:
        """
        Calculate inference data for significant variables across clusters.

        This method calculates inference data for significant variables across clusters, taking into account specified
        conditions and excluding certain columns. It returns a DataFrame with inference results.

        Parameters
        ----------
        return_chi_2_stat : bool, optional
            Whether to include the chi-squared statistic in the output DataFrame. Default is False.

        Returns
        -------
        success : bool
            Indicates whether the calculation was successful and results are available.

        self.deliver_pg_stats: pandas.DataFrame
            Indirectly returned as a class object, derived from results_df.

        Notes
        -----
        The returned DataFrame contains information on significant variables, including their question codes (q_code),
        mode answers (pop_mode), response rates, mode percentages (mode_pop_perc), chi-squared test results (chi_2_result),
        significant categories (sig_more_category), category percentages, and the segments (targeting_seg) they relate to.

        Example
        -------
        success = obj.lazy_calculate_inference_data(return_chi_2_stat=True)
        if success:
            # Access and analyze the inference results.
        """
        results_df = self.inference_schema_df(return_chi_2_stat=return_chi_2_stat)

        variables = self.data.columns.tolist()
        try:
            variables.remove(self.cluster_col)
        except (KeyError, ValueError) as e:
            log.info(
                "lazy_calculate_inference_data - 2. Could not remove cluster_col from variables - column doesn't exist?"
            )

        to_exclude = self.inference_excluded_cols()
        variables = list(set(variables) - set(to_exclude))

        if not self.data[self.cluster_col].empty:
            cluster_labels = np.sort(
                self.data[self.cluster_col].unique().tolist()
            )  # todo: this throws error in python=3.10 ????

            results_df_list = Parallel(n_jobs=-1)(
                delayed(self.generate_lazy_segment_inference)(c, variables)
                for c in cluster_labels
            )

            results_df = pd.concat(results_df_list, axis=0, ignore_index=True)

        else:
            log.info("lazy_calculate_inference_data - could not detect clusters")

        results_df = results_df.sort_values("chi_2_result")

        results_df = results_df.apply(remove_not_select_and_cat_percentage, axis=1)

        # assessing if any significance has been reported
        if results_df.shape[0] == 0:
            pass
        else:
            """removing any empty list entries within sig_more_category from dataframe"""
            results_df = results_df[
                results_df["sig_more_category"].str.len() != 0
                ]

        if results_df.empty:
            log.info("lazy_calculate_inference_data - results_df empty.")
            return False

        self.deliver_pg_stats = results_df
        return True

    def seg_discover_stats_df( # TODO: IS ANYONE USING THIS ?
            self, seg_name, n_feats=10
    ) -> pd.DataFrame:  # todo: rethink at some point the mode retrieval
        """
        Takes top n features (variables) and returns a dataframe with 'q_code', 'mode', 'mode_perc',
        'sig_more_category' of each variable. This means that one variable can have more than one significant
        category.

        Parameters
        ----------
        seg_name: str
            Segment name (cluster name/label)

        n_feats: int, default: 10
            Number of top features/variables to select

        Returns
        -------
        seg_subset: pd.DataFrame
            A dataframe with segment specific top n_feats (sorted by p-value of chi_2_result), see details in notes.

        Notes
        _____

        A dataframe with segment specific top n_feats (sorted by p-value of chi_2_result) with the following variables:
                q_code:
                    this is the question code that has been given when ingested into the class.
                    Data type: string of data code
                mode:
                    This is the mode answer of the question of q_code within the segment, NOT within the population.
                    Data type: String of answer code
                mode_perc:
                    This is the percentage of the segment that has the mode value with the variable (q_code).
                    Data type: Float
                sig_more_category:
                    These are the values (i.e. categories) within the variable (q_code) that have been significantly
                    "more" different to their expected values and influenced the variable (q_code) to be determined as
                    significant.
                    Data type: List of answers, usually they are codes (strings) depending on what has been ingested.

        """
        # produce a subset of the deliver data per segment
        seg_subset = self.deliver_pg_stats[
            self.deliver_pg_stats.targeting_seg == seg_name
            ].copy(deep=True)
        seg_subset = seg_subset.sort_values(by="chi_2_result", ascending=True)
        seg_subset = seg_subset.rename(
            columns={"pop_mode": "mode", "mode_pop_perc": "mode_perc"}
        )

        seg_subset = seg_subset.drop(
            columns=["response_rate", "chi_2_result", "targeting_seg"]
        )
        # removing any significant variables that have mode "not selected" or "not shown"
        seg_subset = seg_subset[seg_subset["mode"] != "not selected"].copy(deep=True)
        seg_subset = seg_subset[seg_subset["mode"] != "Not shown"].copy(deep=True)

        # logging purposes in sentry if error occurs
        feat_codes = seg_subset[:n_feats]["q_code"]

        return seg_subset[:n_feats]

    def return_API_data(self, lazy=True):
        """
        Retrieves inference and discovery data for API consumption.

        Parameters
        ----------
        lazy : bool, optional (default=True)
            If True, calculates inference data using lazy calculation (see methods in class).
            If False, calcualtes all information and laods it into temporary memory.

        Returns
        -------
        deliver_data : pd.DataFrame
            Inference data containing significant variables and other relevant information for the frontend, Sophos,
            clustering metrics...

        discover_data : list
            List of dictionaries, each containing discovery data for individual segments.

        mode_list : list
            List of dictionaries, each containing the mode value for each cluster.

        Notes
        -----
        - The `deliver_data` DataFrame includes significant variables (q_code) and related information.
        - The `discover_data` list contains dictionaries for each segment, with discovery data.
        - The `mode_list` lists the mode value for each cluster.

        """
        if lazy:
            self.lazy_calculate_inference_data()
        else:
            self.extract_deliver_stats_df()

        deliver_data = self.deliver_pg_stats
        # de-encode for the platform
        deliver_data.response_rate = deliver_data.response_rate.astype("float64")

        if self.cluster_col == "cluster":  # todo: this might be better to be relating to the segmentation_type
            try:
                # answer_dict = {y: x for x, y in iter(self.answers.items())}
                # answer_dict['not selected'] = ""
                # answer_dict['not shown'] = ""
                deliver_data['targeting_seg'] = deliver_data['targeting_seg'].map(
                    int)  # todo: also don't think we need this anymore
                # deliver_data = deliver_data.replace(answer_dict)
                deliver_data['targeting_seg'] = deliver_data['targeting_seg'].map(str)
            except:
                log.error("answers from data agent (--> da.get_answers()) have not been passed to API.")  # todo: outdated

        discover_data = []
        for seg_name in np.sort(np.unique(self.data[self.cluster_col].tolist())):
            seg_discover_data = self.seg_discover_stats_df(seg_name=str(seg_name))
            discover_data.append({str(seg_name): seg_discover_data})

        mode_list = []
        clusters = self.data.groupby(self.cluster_col)
        for cluster in clusters:
            mode_dict = {str(cluster[0]): cluster[1].mode(dropna=False).head(1)}
            mode_list.append(mode_dict)
        return deliver_data, discover_data, mode_list


if __name__ == "__main__":
    survey_name = "qudo_consumerzeitgeist_uk_q2_2023_food_test_speed"
    segmentation = None
    clustered_data = pd.read_parquet(
        "s3://qudo-datascience/data-store/surveys/staging/responses_cleaned/qudo_consumerzeitgeist_uk_q2_2023_food_staging/qudo_consumerzeitgeist_uk_q2_2023_food_staging.parquet")
    seg_col = "tech_ww_techcomfort_rb_ord"  # 'QUDO_ALCOHOL_CONSUMER'
    rename_segments = None
