import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pickle
from pathlib import Path
from typing import Tuple
from datetime import datetime
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

#add path path to data folder using sys.path
sys.path.append('../data_process/')
from data_process import load_dataset
from processors.cbc import CBC_REFERENCE_INTERVALS
from processors.outcomes import process_diagnosis

seed = 42
np.random.seed(seed)

def convert_to_datetime(df, unit=None):
    time_cols = [col for col in df.columns if 'time' in col.lower()]
    # dob may be a time col
    if 'dob' in df.columns:
        time_cols.append('dob')
    df[time_cols] = df[time_cols].apply(pd.to_datetime, errors='ignore')
    return df

def get_time_to_event(df, start_col='setpoint_estimation_time', end_col='time_of_death'):
    def calculate_days(row):
        start_time = row[start_col]
        if pd.notna(row[end_col]):
            end_time = row[end_col]
        else:
            end_time = row['observation_period_end_time']
        return (end_time - start_time).days
    
    df['event_status'] = df[end_col].notna().astype(int)
    df['time_to_event'] = df.apply(calculate_days, axis=1)
    return df

def get_age_at_start(df, start_col='setpoint_estimation_time'):
    df['age_at_start'] = (df[start_col] - df['dob']).dt.days / 365.25
    return df

def get_sex(df):
    df['gender_int'] = df['gender'].apply(lambda x: 1 if x == 'F' else 0)
    return df

def merge_setpoints_and_measurements(df, measurements_df, start_time_col='setpoint_estimation_time'):
    # Convert setpoints DataFrame times
    measurements_df['time'] = pd.to_datetime(measurements_df['time'], format='%Y-%m-%d')

    # Now perform the merge_asof with compatible datetime types
    merged_df = pd.merge_asof(
        df.sort_values(start_time_col),
        measurements_df.sort_values('time'),
        left_on=start_time_col,
        right_on="time",
        by="subject_id",
        direction="nearest"
    )
    
    # can you create pd.dummies variable from BMI_category where the refernce group is normal weight?
    merged_df['BMI_category'] = pd.Categorical(merged_df['BMI_category'])
    merged_df = pd.get_dummies(merged_df, columns=['BMI_category'])
    merged_df.drop(columns=['BMI_category_Normal'], inplace=True)
 
    
    return merged_df

def value_in_reference_interval(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    def is_in_interval(row):
        interval = CBC_REFERENCE_INTERVALS[row['code']][row['gender']]
        midpoint = (interval[0] + interval[1]) / 2
        value = row[value_col]
        bool_in_interval = interval[0] <= value <= interval[1]
        abs_diff = abs(value - midpoint)
        return bool_in_interval, abs_diff
    
    if value_col == 'setpoint':
        df[f'{value_col}_in_interval'], df[f'diff_{value_col}_reference'] = zip(*df.apply(is_in_interval, axis=1))
    else:
        df['within_reference'], df['diff_to_reference'] = zip(*df.apply(is_in_interval, axis=1))
        df['not_within_reference'] = ~df['within_reference']
    return df

def value_in_setpoint_interval(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    def is_in_interval(row):
        interval = (row['setpoint'] - row['cv'] * 2, row['setpoint'] + row['cv'] * 2)
        midpoint = (interval[0] + interval[1]) / 2
        value = row[value_col]
        bool_in_interval = interval[0] <= value <= interval[1]
        abs_diff = abs(value - midpoint)
        return bool_in_interval, abs_diff
    
    df['within_setpoint'], df['diff_to_setpoint'] = zip(*df.apply(is_in_interval, axis=1))
    df['not_within_setpoint'] = ~df['within_setpoint']
    return df

def normalize_setpoint(df: pd.DataFrame) -> pd.DataFrame:
    df_list = []
    
    for code in CBC_REFERENCE_INTERVALS.keys():
        sub_df = df[df['code'] == code].copy()
        
        sub_df['cv_normalized'] = sub_df['cv']
        sub_df['setpoint_normalized'] = sub_df['setpoint'] / sub_df['cv'].mean()
        df_list.append(sub_df)
    
    return pd.concat(df_list, ignore_index=True)

def split_train_test(df, code):
    df = df[df['code'] == code]
    patients = df['subject_id'].unique()
    np.random.shuffle(patients)
    split_index = int(len(patients) * 0.8)
    train_patients = patients[:split_index]
    test_patients = patients[split_index:]
    train_set = df[df['subject_id'].isin(train_patients)]
    test_set = df[df['subject_id'].isin(test_patients)]
    #print(f"Code: {code}, Train set size: {len(train_set)}, Test set size: {len(test_set)}")
    return train_set, test_set

def process_dataframe(dfs, args):
    min_days = args['min_days']
    min_test = args['min_test']
    year_cutoff = args['year_cutoff']
    min_time = args['min_time']
    max_time = args['max_time']
    event_col = args['event_col']
    setpoints = pd.read_csv(f'../results/setpoint_calculations/setpoints_gap:{min_days}_tests:{min_test}_year:{year_cutoff}.csv')
    df = (setpoints
            .merge(dfs['demo_df'], on='subject_id', how='left')
            .merge(dfs['mortality_df'], on='subject_id', how='left')
            .merge(dfs['cbc_df'], on=['subject_id', 'code'], how='left')).dropna(subset=['setpoint_estimation_time'])
    df = convert_to_datetime(df)
    df = merge_setpoints_and_measurements(df, dfs['measurements_df'], 'time')
    df = value_in_setpoint_interval(df, 'numeric_value')
    df = value_in_reference_interval(df, 'numeric_value')
    df = value_in_reference_interval(df, 'setpoint')
    df = get_time_to_event(df, start_col='time', end_col=event_col)
    df = get_age_at_start(df, 'time')
    df = get_sex(df)
    
    df['time_since_setpoint'] = (df['time'] - df['setpoint_estimation_time']).dt.days
    df = df[df['time_since_setpoint'] >= min_time].sort_values(by='time_since_setpoint')
    df = df[df['time_since_setpoint'] <= max_time].sort_values(by='time_since_setpoint')

    # depending on the group,  lowest (HCT, HGB, MCH, MCV, MCHC, PLT, RBC) or highest (RDW, WBC)
    min_group = ['HCT', 'HGB', 'MCH', 'MCV', 'MCHC', 'PLT', 'RBC']
    max_group = ['RDW', 'WBC']

    # For min group, get the row with minimum value for each subject_id and code
    df_min = df[df['code'].isin(min_group)].sort_values('numeric_value').groupby(['subject_id', 'code']).first().reset_index()
    df_max = df[df['code'].isin(max_group)].sort_values('numeric_value', ascending=False).groupby(['subject_id', 'code']).first().reset_index()
    df = pd.concat([df_min, df_max])
    return df 

def evaluate_cox_model(event_probabilities, code, biomarker, n_bootstrap=1000, alpha=0.95):
    """
    Calculate concordance index with bootstrap confidence intervals for each prediction time.
    """
    results = []
    for (time, biomarker), group in event_probabilities.groupby(["prediction_time", "biomarker"]):
        # Calculate concordance on full dataset
        try:
            c_index = concordance_index(
                group["time_to_event"],
                1-group["event_probability"],
                group["event_status"]
            )
        except ZeroDivisionError:
            print(f"Warning: No admissible pairs for {time} year prediction, {biomarker}")
            c_index = np.nan
        
        # Bootstrap
        bootstrap_samples = []
        n_bootstrap = int(n_bootstrap)
        for _ in range(n_bootstrap):
            # Sample with replacement
            boot_group = group.sample(n=len(group), replace=True)
            try:
                boot_c_index = concordance_index(
                    boot_group["time_to_event"],
                    1-boot_group["event_probability"],
                    boot_group["event_status"]
                )
                bootstrap_samples.append(boot_c_index)
            except ZeroDivisionError:
                continue
        
        # Calculate confidence intervals only if we have valid bootstrap samples
        if bootstrap_samples:
            ci_lower, ci_upper = np.percentile(bootstrap_samples, 
                                             [(1 - alpha) * 100/2, 
                                              (1 + alpha) * 100/2])
            ci_mean = np.mean(bootstrap_samples)
        else:
            ci_mean, ci_lower, ci_upper = np.nan, np.nan, np.nan
        
        results.append({
            "biomarker": biomarker,
            "code": code,
            "prediction_time": time,
            "c_index": ci_mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })
        
    results = pd.DataFrame(results)
    return results

def run_cox_model(df, code, biomarker='setpoint', covariates=[], split=False, time_points=[3, 5, 10], n_bootstrap=1000):
    features = [biomarker] + covariates + ['time_to_event', 'event_status']
    results = {
        "Code": code,
        "Biomarker": biomarker,
        "N": len(df)
    }
    
    if split:
        # Use the existing split_train_test function
        train_df, test_df = split_train_test(df, code)
        cph = CoxPHFitter()
        cph.fit(train_df[features], duration_col='time_to_event', event_col='event_status')
        
        # Get metrics for training data
        hr = cph.hazard_ratios_
        summary = cph.summary
        
        results.update({
            "HR": hr[biomarker],
            "LOWER CI": summary.loc[biomarker, 'exp(coef) lower 95%'],
            "UPPER CI": summary.loc[biomarker, 'exp(coef) upper 95%'],
            "p-value": summary.loc[biomarker, "p"],
            "train_concordance": cph.score(train_df[features], scoring_method="concordance_index"),
            #"test_concordance": cph.score(test_df[features], scoring_method="concordance_index"),
            "N_train": len(train_df),
            "N_test": len(test_df),
            "positive_train": len(train_df[train_df['event_status'] == 1]),
            "positive_test": len(test_df[test_df['event_status'] == 1]),
            # "1year_pred_cstat": c_index_df.loc[biomarker, '1y_pred_cstat'],
            # "3year_pred_cstat": c_index_df.loc[biomarker, '3y_pred_cstat'],
            # "5year_pred_cstat": c_index_df.loc[biomarker, '5y_pred_cstat'],
            # "10year_pred_cstat": c_index_df.loc[biomarker, '10y_pred_cstat']
        })
        
        event_probabilities = get_event_probabilities(cph, test_df, time_points, biomarker)
        metrics = evaluate_cox_model(event_probabilities, code, biomarker, n_bootstrap)
        
    else:
        # Original behavior without train/test split
        metrics = None
        cph = CoxPHFitter()
        cph.fit(df[features], duration_col='time_to_event', event_col='event_status')
        hr = cph.hazard_ratios_
        summary = cph.summary
        results.update({
            "HR": hr[biomarker],
            "LOWER CI": summary.loc[biomarker, 'exp(coef) lower 95%'],
            "UPPER CI": summary.loc[biomarker, 'exp(coef) upper 95%'],
            "p-value": summary.loc[biomarker, "p"],
            "concordance": cph.score(df[features], scoring_method="concordance_index")
        })
    
    return results, metrics

def get_event_probabilities(cph, df, time_points, biomarker):
    features = cph.summary.index.tolist()
    survival_probabilities = cph.predict_survival_function(df[features], times=time_points)
    event_probabilities = 1 - survival_probabilities
    df_event_probabilities = pd.DataFrame(event_probabilities.T, columns=time_points)
    df_event_probabilities['subject_id'] = df['subject_id']
    df_event_probabilities = df_event_probabilities.melt(id_vars=['subject_id'], var_name='prediction_time', value_name='event_probability')
    df_event_probabilities['biomarker'] = biomarker
    df_event_probabilities = df_event_probabilities.merge(df[['subject_id', 'time_to_event', 'event_status']], on='subject_id', how='left')

    return df_event_probabilities


def run_cox_analysis(df, min_days, min_test, year_cutoff, features=[], split=False):
      df = convert_to_datetime(df)
      df = get_time_to_event(df, 'setpoint_estimation_time') 
      df = get_age_at_start(df, 'setpoint_estimation_time')
      df = get_sex(df)
      df = normalize_setpoint(df)
      df = value_in_reference_interval(df, 'setpoint')
      
      results = []
      metrics = []
      results_reference = []
      metrics_reference = []
      features = ['gender_int', 'age_at_start'] + features
      for code in CBC_REFERENCE_INTERVALS.keys():
            df_sub = df[df['code'] == code]
            df_sub_reference = df_sub[df_sub['setpoint_in_interval'] == True]
            if len(df_sub) > 0 and code != 'WBC':
                  results_setpoint, metrics_setpoint = run_cox_model(df_sub, code, biomarker='setpoint_normalized', covariates=features, split=split)
                  results_cv, metrics_cv = run_cox_model(df_sub, code, biomarker='cv_normalized', covariates=features, split=split)
                  results.append(results_setpoint)
                  results.append(results_cv)
                  metrics.extend(metrics_setpoint)
                  metrics.extend(metrics_cv)
                  
                  results_setpoint_reference, metrics_setpoint_reference = run_cox_model(df_sub_reference, code, biomarker='setpoint_normalized', covariates=features, split=split)
                  results_cv_reference, metrics_cv_reference = run_cox_model(df_sub_reference, code, biomarker='cv_normalized', covariates=features, split=split)
                  results_reference.append(results_setpoint_reference)
                  results_reference.append(results_cv_reference)
                  metrics_reference.extend(metrics_setpoint_reference)
                  metrics_reference.extend(metrics_cv_reference)

      results_df = pd.DataFrame(results)
      results_reference_df = pd.DataFrame(results_reference)
      
      return results_df, results_reference_df

def run_interval_cox_analysis(df, biomarker_type='bool', features=[], split=False, setpoint_in_interval=False, n_bootstrap=1000):

    features = ['gender_int', 'age_at_start'] + features
    results = []
    metrics_df = pd.DataFrame()
    for code in CBC_REFERENCE_INTERVALS.keys():
        df_sub = df[df['code'] == code]
        if setpoint_in_interval:
            df_sub = df_sub[df_sub['setpoint_in_interval'] == True]
        if len(df_sub) > 0 and code != 'WBC':
            if biomarker_type == 'bool':
                results_not_within_setpoint, metrics_not_within_setpoint = run_cox_model(df_sub, code, biomarker='not_within_setpoint', covariates=features, split=split, n_bootstrap=n_bootstrap)            
                results_not_within_reference, metrics_not_within_reference = run_cox_model(df_sub, code, biomarker='not_within_reference', covariates=features, split=split, n_bootstrap=n_bootstrap)
            elif biomarker_type == 'numeric':
                results_not_within_setpoint, metrics_not_within_setpoint = run_cox_model(df_sub, code, biomarker='diff_to_setpoint', covariates=features, split=split, n_bootstrap=n_bootstrap)            
                results_not_within_reference, metrics_not_within_reference = run_cox_model(df_sub, code, biomarker='diff_to_reference', covariates=features, split=split, n_bootstrap=n_bootstrap)
            else:
                raise ValueError(f"Invalid biomarker type: {biomarker_type}")
            results.append(results_not_within_setpoint)
            results.append(results_not_within_reference)
            metrics_df = pd.concat([metrics_df, metrics_not_within_setpoint])
            metrics_df = pd.concat([metrics_df, metrics_not_within_reference])
        
    results_df = pd.DataFrame(results)
    metrics_df = metrics_df 
    return results_df, metrics_df 

    
def merge_diagnosis_with_setpoints(df, diag_df, start_time_col='setpoint_estimation_time'):
    # First check and fix any duplicate columns
    diag_df = diag_df.loc[:, ~diag_df.columns.duplicated()]
    df[start_time_col] = pd.to_datetime(df[start_time_col])
    diag_df['diagnosis_time'] = pd.to_datetime(diag_df['diagnosis_time'])
    # Sort both dataframes by their respective time columns
    df_sorted = df.sort_values(start_time_col)
    diag_df_sorted = diag_df.sort_values('diagnosis_time')

    # merge with the diagnosis data
    df = pd.merge_asof(
        df_sorted,
        diag_df_sorted,
        left_on=start_time_col,
        right_on='diagnosis_time',
        by="subject_id",
        direction='backward'
    ).fillna(0)
    
    # Drop the diagnosis_time column
    df = df.drop(columns=['diagnosis_time'])
    
    return df
