# This file can take all the raw measurements and then apply data filtering in the following ways
# 1. Foy method (used isolated CBCs)
# 2. Yash method (use longest spaced subsequence)
# 3. Cutoff method (use 80% of total timeframe)

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import sys, os

sys.path.append( '../')
from process.config import LOINC_CODES, REVERSE_LOINC_CODES, DATAPATH
import logging
import os
from tqdm import tqdm

def filter_measurements_foy(df, min_gap=30, min_tests=5, index_year=2018):
    """
    Filters DataFrame for (subject_id, test_name) groups that have at least `min_tests`
    measurements before `index_year`, each at least `min_gap_days` apart.
    Retains those and all post-`index_year` measurements from qualifying groups.
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    isolated_rows = []
    grouped = df.groupby(['subject_id', 'test_name'])

    for (pt, test), group in tqdm(grouped, desc="Foy Filter"):
        pre_group = group[group['time'] < pd.Timestamp(f'{index_year}-01-01')]

        if len(pre_group) < min_tests:
            continue

        # Sort by date
        sorted_group = pre_group.sort_values('time')
        sorted_dates = sorted_group['time'].values
        sorted_indices = sorted_group.index

        isolated_indices = []

        for i in range(len(sorted_dates)):
            left_ok = (i == 0) or (sorted_dates[i] - sorted_dates[i - 1] > np.timedelta64(min_gap, 'D'))
            right_ok = (i == len(sorted_dates) - 1) or (sorted_dates[i + 1] - sorted_dates[i] > np.timedelta64(min_gap, 'D'))

            if left_ok and right_ok:
                isolated_indices.append(sorted_indices[i])

        if len(isolated_indices) >= min_tests:
            isolated_rows.extend(isolated_indices)

    df2 = df.loc[np.unique(isolated_rows)]
    return df2

def filter_measurements_yash(df, min_gap=30, min_tests=5, index_year=2018):
    grouped = df.groupby(['subject_id', 'test_name'])   
    df['time'] = pd.to_datetime(df['time'])

    # i find the longest subsequence of tests before the index year
    # i check the subseq is at least 5 long and there is at least 1 more test post index
    rows_to_keep = []
    for (pt, test), group in tqdm(grouped, desc="Yash Filter"):
            pre_group = group[group['time'] < pd.Timestamp(f'{index_year}-01-01')]

            # Sort pre_group by time to get ordered dates with preserved index
            pre_group_sorted = pre_group.sort_values('time')
            dates = pre_group_sorted['time'].values

            if len(dates) < min_tests:
                continue  # skip if not enough test entries

            # Build the subsequence and track corresponding indices
            subsequence_indices = [pre_group_sorted.index[0]]
            last_date = dates[0]

            for i in range(1, len(dates)):
                if (dates[i] - last_date) / np.timedelta64(1, 'D') >= min_gap:
                    subsequence_indices.append(pre_group_sorted.index[i])
                    last_date = dates[i]

            # Keep if at least 5 tests are in subsequence
            if len(subsequence_indices) >= min_tests:
                rows_to_keep.extend(subsequence_indices)

    df2 = df.loc[np.unique(rows_to_keep)]
    return df2

def filter_measurements_cutoff(df, percent=80, min_tests=5):
    percent = percent/100
    df['time'] = pd.to_datetime(df['time'])
    grouped = df.groupby(['subject_id', 'test_name'])
    rows_to_keep = []

    for (pt, test), group in tqdm(grouped, desc="Cutoff Filter"):
        # Sort by time
        group_sorted = group.sort_values('time')
        if len(group_sorted) < min_tests:
            continue

        # Get time range
        start_time = group_sorted['time'].min()
        end_time = group_sorted['time'].max()
        total_time_range = end_time - start_time

        # Calculate cutoff time by converting timedelta to hours first
        time = total_time_range.total_seconds() 
        cutoff = float(time) * percent
        
        cutoff_time = start_time + pd.Timedelta(seconds=cutoff)
        measurements_in_range = group_sorted[group_sorted['time'] <= cutoff_time]
        measurements_post_range = group_sorted[group_sorted['time'] > cutoff_time]
        
        if len(measurements_in_range) >= min_tests and len(measurements_post_range) > 0:
            rows_to_keep.extend(measurements_in_range.index)
        
    df_filtered = df.loc[np.unique(rows_to_keep)]

    return df_filtered

if __name__=="__main__":
    all_meas_df = pd.read_pickle("../data/processed/all_lab_measurements.pkl")

    labs_filter_foy = filter_measurements_foy(all_meas_df, 30, 5, 2018)
    labs_filter_foy.to_pickle("../data/filter/labs_filter_foy_30_5_2018.pkl")

    labs_filter_yash = filter_measurements_yash(all_meas_df, 30, 5, 2018)
    labs_filter_yash.to_pickle("../data/filter/labs_filter_yash_30_5_2018.pkl")

    labs_filter_cutoff = filter_measurements_cutoff(all_meas_df, 80, 5)
    labs_filter_cutoff.to_pickle("../data/filter/labs_filter_cutoff_80_5.pkl")