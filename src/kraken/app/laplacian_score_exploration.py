### Exploration to see how optimal n_neighbours changes with sample size

import math

import numpy as np
import pandas as pd
from sklearn.utils import resample

from app.feature_selection.feature_selection import get_optimal_laplacian

# fin_ser = 's3://qudo-datascience/data-store/staging_surveys/surveys_cleaned/qudo_financialservicesfinal_uk_q1_2023_staging/qudo_financialservicesfinal_uk_q1_2023_staging_responses/'
# fin_ser_demo = 's3://qudo-datascience/data-store/staging_surveys/surveys_cleaned/qudo_financialservices_uk_q1_2023/qudo_financialservices_uk_q1_2023_responses/qudo_financialservices_uk_q1_2023_responses.parquet'

rdf = pd.read_parquet('data/qudo_financialservicesfinal_uk_q1_2023_staging_responses.parquet')


def subsamping_eval(df, start, end, n_cuts):
    proportion = 1 / n_cuts

    sample_sizes = len(df) * np.arange(start, end + proportion, proportion)
    if start != 0:
        n_samples = [math.floor(x) for x in sample_sizes]
    else:
        n_samples = [math.floor(x) for x in sample_sizes][1:]
    outcome_list = []
    for i in n_samples:
        sample_df = resample(df, n_samples=i, replace=False)
        results = get_optimal_laplacian(sample_df)
        results['sample_size'] = i
        outcome_list.append(results)

    outcome_df = pd.DataFrame(outcome_list)
    return outcome_df


a = subsamping_eval(rdf, 0.5, 0.8, 10)
