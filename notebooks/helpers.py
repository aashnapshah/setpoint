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

# Plot styling configuration
PLOT_STYLE = {
    'figure.dpi': 150,
    'font.size': 6,
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'axes.labelsize': 6,  
    'legend.fontsize': 6,
    'axes.titlesize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'axes.edgecolor': 'k',
    'axes.linewidth': 0.5,
    'axes.grid': False,
    'axes.prop_cycle': plt.cycler(color=sns.husl_palette(h=.7)),
    'figure.figsize': (2, 2),
    'xtick.major.pad': -3,
    'ytick.major.pad': -3
}

plt.rcParams.update(PLOT_STYLE)
sns.set_theme(style="white", font='Arial', rc=PLOT_STYLE)

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
        value = row[value_col]
        return interval[0] <= value <= interval[1]
    df[f'{value_col}_in_interval'] = df.apply(is_in_interval, axis=1)
    return df

def value_in_setpoint_interval(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    def is_in_interval(row):
        interval = (row['setpoint'] - row['cv'] * 2, row['setpoint'] + row['cv'] * 2)
        value = row[value_col]
        return interval[0] <= value <= interval[1]
    df[f'within_setpoint'] = df.apply(is_in_interval, axis=1)
    return df

def normalize_setpoint(df: pd.DataFrame) -> pd.DataFrame:
    df_list = []
    
    for code in CBC_REFERENCE_INTERVALS.keys():
        sub_df = df[df['code'] == code].copy()
        
        sub_df['cv_normalized'] = sub_df['cv']
        sub_df['setpoint_normalized'] = sub_df['setpoint'] / sub_df['cv'].mean()
        df_list.append(sub_df)
    
    return pd.concat(df_list, ignore_index=True)


def form_train_test_set(df, min_time_since_setpoint, max_time_since_setpoint):
    df_ = value_in_setpoint_interval(df, 'numeric_value')
    df_ = value_in_reference_interval(df_, 'setpoint')
    df_ = get_time_to_event(df_, 'time')
    df_ = get_age_at_start(df_, 'time')
    df_ = get_sex(df_)

    df_['not_within_setpoint'] = ~df_['within_setpoint']
    df_['not_within_reference'] = ~df_['within_reference']
    df_['time_since_setpoint'] = (df_['time'] - df_['setpoint_estimation_time']).dt.days
    df_ = df_[df_['time_since_setpoint'] >= min_time_since_setpoint].sort_values(by='time_since_setpoint')
    df_ = df_[df_['time_since_setpoint'] <= max_time_since_setpoint].sort_values(by='time_since_setpoint')

    # depending on the group,  lowest (HCT, HGB, MCH, MCV, MCHC, PLT, RBC) or highest (RDW, WBC)
    #df = df.groupby(['subject_id', 'code']).first().reset_index()
    # Define the groups for min/max selection
    min_group = ['HCT', 'HGB', 'MCH', 'MCV', 'MCHC', 'PLT', 'RBC']
    max_group = ['RDW', 'WBC']

    # For min group, get the row with minimum value for each subject_id and code
    df_min = df_[df_['code'].isin(min_group)].sort_values('numeric_value').groupby(['subject_id', 'code']).first().reset_index()
    df_max = df_[df_['code'].isin(max_group)].sort_values('numeric_value', ascending=False).groupby(['subject_id', 'code']).first().reset_index()

    # Combine the results
    df_ = pd.concat([df_min, df_max])
    return df_

def split_train_test(df, code):
    df = df[df['code'] == code]
    # make sure no patient has both a training and a test set and then randomly split the patients into a training and a test set
    patients = df['subject_id'].unique()
    np.random.shuffle(patients)
    split_index = int(len(patients) * 0.8)
    train_patients = patients[:split_index]
    test_patients = patients[split_index:]
    train_set = df[df['subject_id'].isin(train_patients)]
    test_set = df[df['subject_id'].isin(test_patients)]
    #train_set = form_train_test_set(train_set)
    #test_set = form_train_test_set(test_set)
    return train_set, test_set

def evaluate_cox_model(event_probabilities, code, biomarker, n_bootstrap=1000, alpha=0.95):
    
    results = []
    for (time, biomarker), group in event_probabilities.groupby(["prediction_time", "biomarker"]):
        # Calculate concordance on full dataset
        c_index = concordance_index(
            group["time_to_event"],
            1-group["event_probability"],
            group["event_status"]
        )
        
        # Bootstrap
        bootstrap_samples = []
        n_bootstrap = int(n_bootstrap)  # Convert to integer
        for _ in range(n_bootstrap):
            # Sample with replacement
            boot_group = group.sample(n=len(group), replace=True)
            boot_c_index = concordance_index(
                boot_group["time_to_event"],
                1-boot_group["event_probability"],
                boot_group["event_status"]
            )
            bootstrap_samples.append(boot_c_index)
        
        # Calculate confidence intervals
        ci_lower, ci_upper = np.percentile(bootstrap_samples, 
                                            [(1 - alpha) * 100/2, 
                                            (1 + alpha) * 100/2])
        
        results.append({
            "biomarker": biomarker,
            "code": code,
            "prediction_time": time,
            "c_index": c_index,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })
        
    results = pd.DataFrame(results)
    return results

def run_cox_model(df, code, biomarker='setpoint', covariates=[], use_train_test=False, time_points=[3, 5, 10]):
    features = [biomarker] + covariates + ['time_to_event', 'event_status']
    results = {
        "Code": code,
        "Biomarker": biomarker,
        "N": len(df)
    }
    
    if use_train_test:
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
            "test_concordance": cph.score(test_df[features], scoring_method="concordance_index"),
            "N_train": len(train_df),
            "N_test": len(test_df)
            # "1year_pred_cstat": c_index_df.loc[biomarker, '1y_pred_cstat'],
            # "3year_pred_cstat": c_index_df.loc[biomarker, '3y_pred_cstat'],
            # "5year_pred_cstat": c_index_df.loc[biomarker, '5y_pred_cstat'],
            # "10year_pred_cstat": c_index_df.loc[biomarker, '10y_pred_cstat']
        })
        
        event_probabilities = get_event_probabilities(cph, test_df, time_points, biomarker)
        metrics = evaluate_cox_model(event_probabilities, code, biomarker)
        
    else:
        # Original behavior without train/test split
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

def plot_setpoint_hr(results, title):
    # Create the scatter plot
    results = results.sort_values(by='Code')
    results['Biomarker'] = results['Biomarker'].str.split('_').str[0].str.capitalize()
    results['Code'] = results['Code'] + ' (N=' + results['N'].astype(str) + ')'
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(4, 2))
    # want to plot the HR for setpoint on the left and the HR for cv on the right
    results_setpoint = results[results['Biomarker'] == 'Setpoint']
    results_cv = results[results['Biomarker'] == 'Cv']
    
    sns.scatterplot(data=results_setpoint, x='HR', y='Code', ax=ax[0])
    sns.scatterplot(data=results_cv, x='HR', y='Code', ax=ax[1])
    
    ax[0].errorbar(x=results_setpoint['HR'], y=range(len(results_setpoint)), 
                xerr=[results_setpoint['HR'] - results_setpoint['LOWER CI'], 
                    results_setpoint['UPPER CI'] - results_setpoint['HR']],
                fmt='none', c='gray', alpha=0.5, capsize=3)
    ax[0].set_xlabel('HR')
    ax[0].set_ylabel('')
    ax[0].set_title('Setpoint')
    ax[0].axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax[0].set_xlim(-0.1, 2.2)
    
    ax[1].errorbar(x=results_cv['HR'], y=range(len(results_cv)), 
                xerr=[results_cv['HR'] - results_cv['LOWER CI'], 
                    results_cv['UPPER CI'] - results_cv['HR']],
                fmt='none', c='gray', alpha=0.5, capsize=3)
    ax[1].set_xlabel('HR')
    ax[1].set_ylabel('')
    ax[1].set_title('CV')
    ax[1].axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax[1].set_xlim(0.8, 1.4)
    
    plt.suptitle(title, fontsize=6)
    plt.tight_layout()
    return plt

def run_cox_analysis(df, min_days, min_test, year_cutoff, features=[], use_train_test=False):
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
                  results_setpoint, metrics_setpoint = run_cox_model(df_sub, code, biomarker='setpoint_normalized', covariates=features, use_train_test=use_train_test)
                  results_cv, metrics_cv = run_cox_model(df_sub, code, biomarker='cv_normalized', covariates=features, use_train_test=use_train_test)
                  results.append(results_setpoint)
                  results.append(results_cv)
                  metrics.extend(metrics_setpoint)
                  metrics.extend(metrics_cv)
                  
                  results_setpoint_reference, metrics_setpoint_reference = run_cox_model(df_sub_reference, code, biomarker='setpoint_normalized', covariates=features, use_train_test=use_train_test)
                  results_cv_reference, metrics_cv_reference = run_cox_model(df_sub_reference, code, biomarker='cv_normalized', covariates=features, use_train_test=use_train_test)
                  results_reference.append(results_setpoint_reference)
                  results_reference.append(results_cv_reference)
                  metrics_reference.extend(metrics_setpoint_reference)
                  metrics_reference.extend(metrics_cv_reference)

      results_df = pd.DataFrame(results)
      results_reference_df = pd.DataFrame(results_reference)
      
      return results_df, results_reference_df

def run_interval_cox_analysis(df, min_time_since_setpoint, max_time_since_setpoint, features=[], use_train_test=False, setpoint_in_interval=False):
    df = value_in_setpoint_interval(df, 'numeric_value')
    df = value_in_reference_interval(df, 'setpoint')
    df = get_time_to_event(df, 'time')
    df = get_age_at_start(df, 'time')
    df = get_sex(df)

    df['not_within_setpoint'] = ~df['within_setpoint']
    df['not_within_reference'] = ~df['within_reference']
    df['time_since_setpoint'] = (df['time'] - df['setpoint_estimation_time']).dt.days
    df = df[df['time_since_setpoint'] >= min_time_since_setpoint].sort_values(by='time_since_setpoint')
    df = df[df['time_since_setpoint'] <= max_time_since_setpoint].sort_values(by='time_since_setpoint')

    # depending on the group,  lowest (HCT, HGB, MCH, MCV, MCHC, PLT, RBC) or highest (RDW, WBC)
    min_group = ['HCT', 'HGB', 'MCH', 'MCV', 'MCHC', 'PLT', 'RBC']
    max_group = ['RDW', 'WBC']

    # For min group, get the row with minimum value for each subject_id and code
    df_min = df[df['code'].isin(min_group)].sort_values('numeric_value').groupby(['subject_id', 'code']).first().reset_index()
    df_max = df[df['code'].isin(max_group)].sort_values('numeric_value', ascending=False).groupby(['subject_id', 'code']).first().reset_index()

    # Combine the results
    df = pd.concat([df_min, df_max])

    features = ['gender_int', 'age_at_start'] + features
    results = []
    metrics_df = pd.DataFrame()
    for code in CBC_REFERENCE_INTERVALS.keys():
        df_sub = df[df['code'] == code]
        if setpoint_in_interval:
            df_sub = df_sub[df_sub['setpoint_in_interval'] == True]
        if len(df_sub) > 0 and code != 'WBC':
            results_not_within_setpoint, metrics_not_within_setpoint = run_cox_model(df_sub, code, biomarker='not_within_setpoint', covariates=features, use_train_test=use_train_test)            
            results_not_within_reference, metrics_not_within_reference = run_cox_model(df_sub, code, biomarker='not_within_reference', covariates=features, use_train_test=use_train_test)
            
            results.append(results_not_within_setpoint)
            results.append(results_not_within_reference)
            metrics_df = pd.concat([metrics_df, metrics_not_within_setpoint])
            metrics_df = pd.concat([metrics_df, metrics_not_within_reference])
        
    results_df = pd.DataFrame(results)
    metrics_df = metrics_df #[['Code', 'Biomarker', 'prediction_time', 'c_index']]
    return results_df, metrics_df 

def plot_interval_hr(results, title):
    # make a deep copy of the results dataframe
    results = results.copy(deep=True)
    results['Biomarker'] = results['Biomarker'].str.split('_').str[2].str.capitalize()
    
    plt.figure(figsize=(4, 3))
    palette = sns.husl_palette(h=.7)
    colors = {'Setpoint': 'orange', 'Reference': palette[0]}
    sns.scatterplot(
        data=results, 
        x='HR', 
        y='Code', 
        hue='Biomarker',
        palette=colors,
        s=50,  
        alpha=0.7
    )

    for i, type_name in enumerate(results['Biomarker'].unique()):
        type_df = results[results['Biomarker'] == type_name]
        plt.errorbar(
            x=type_df['HR'], 
            y=type_df.Code,
            xerr=[type_df['HR'] - type_df['LOWER CI'], 
                type_df['UPPER CI'] - type_df['HR']],
            fmt='none', 
            c=colors[type_name],
            alpha=0.3,
            capsize=3,
            capthick=1,
            elinewidth=1
        )

    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    plt.xlabel('Hazard Ratio (95% CI)', fontsize=8)
    plt.ylabel('CBC Measure', fontsize=8)

    plt.legend(
        #title=title,
        title_fontsize=8,
        fontsize=7,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    plt.title(title)
    plt.tight_layout()
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.show()
    
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

def plot_interval_concordance(results, title):
    # Create subplots for each prediction time
    prediction_times = results['prediction_time'].unique()

    fig, axes = plt.subplots(1, len(prediction_times), figsize=(15, 3), sharey=True, sharex=True)
    axes = axes.ravel()
    
    # Plot for each prediction time
    for idx, time in enumerate(prediction_times):
        time_data = results[results['prediction_time'] == time]
        time_data['Biomarker'] = time_data['biomarker'].str.split('_').str[2].str.capitalize()
        
        palette = sns.husl_palette(h=.7)
        colors = {'Setpoint': 'orange', 'Reference': palette[0]}
        
        # Create scatter plot
        sns.scatterplot(
            data=time_data, 
            x='c_index', 
            y='code', 
            hue='Biomarker',
            palette=colors,
            s=50,  
            alpha=0.7,
            ax=axes[idx]
        )

        # Add error bars for each biomarker
        for i, type_name in enumerate(time_data['Biomarker'].unique()):
            type_df = time_data[time_data['Biomarker'] == type_name]
            axes[idx].errorbar(
                x=type_df['c_index'], 
                y=type_df.code,
                xerr=[type_df['c_index'] - type_df['ci_lower'], 
                    type_df['ci_upper'] - type_df['c_index']],
                fmt='none', 
                c=colors[type_name],
                alpha=0.3,
                capsize=3,
                capthick=1,
                elinewidth=1
            )

        axes[idx].set_xlabel('Concordance Index (95% CI)', fontsize=8)
        axes[idx].set_ylabel('CBC Measure', fontsize=8)
        axes[idx].set_title(f'{time} Year Prediction')
        
        # Remove individual legends except for the last subplot
        if idx < 3:
            axes[idx].get_legend().remove()
        
        axes[idx].tick_params(axis='both', labelsize=7)

    # Adjust the legend position for the last subplot
    axes[-1].legend(
        title=title,
        title_fontsize=8,
        fontsize=7,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    plt.tight_layout()
    return fig, axes