import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
import seaborn as sns
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import pickle

sys.path.append('../')

from process.config import REFERENCE_INTERVALS, MIN, MAX

def create_simulated_outcomes(patient_data, lab_data):
    np.random.seed(42)
    unique_patients = patient_data['subject_id'].unique()
    random_dates = pd.date_range(start='2018-01-01', end='2025-12-31')

    outcomes_df = pd.DataFrame({
        'subject_id': unique_patients,
        'mortality_date': np.where(
            np.random.random(len(unique_patients)) < 0.5,
            [np.random.choice(random_dates) for _ in range(len(unique_patients))],
            pd.NA
        )
    })

    outcomes_df['mortality_date'] = pd.to_datetime(outcomes_df['mortality_date'])
    outcomes_df['censorship_date'] = outcomes_df.apply(
        lambda x: x.mortality_date if pd.notna(x.mortality_date)
        else np.random.choice(random_dates),
        axis=1
    )

    return outcomes_df

def convert_to_category(data, column_name):
    data[column_name] = data[column_name].astype('category').cat.codes
    return data

def extract_followup_labs(reference_data, lab_measurements, min_days=10, max_days=365*2):
    reference_data = reference_data.copy()
    reference_data['setpoint_date'] = pd.to_datetime(reference_data['setpoint_date'])
    reference_data.drop(columns=['age'], inplace=True)

    lab_measurements['time'] = pd.to_datetime(lab_measurements['time'])
    reference_data = reference_data.merge(
        lab_measurements[['subject_id', 'test_name', 'numeric_value', 'time', 'age']], 
        on=['subject_id', 'test_name'],
        how='left'
    )
    
    time_window_mask = (
        (reference_data['time'] > reference_data['setpoint_date'] + pd.Timedelta(days=min_days)) & 
        (reference_data['time'] <= reference_data['setpoint_date'] + pd.Timedelta(days=max_days))
    )
    reference_data = reference_data[time_window_mask]

    min_values = (reference_data[reference_data['test_name'].isin(MIN)]
             .groupby(['subject_id', 'test_name'])
             .min()
             .reset_index())
             
    max_values = (reference_data[reference_data['test_name'].isin(MAX)]
             .groupby(['subject_id', 'test_name'])
             .max()
             .reset_index())

    drop_cols = ['setpoint_date', 'time_period_days', 'num_measurements', 'w']
    min_values = min_values.drop(columns=drop_cols)
    max_values = max_values.drop(columns=drop_cols)
    
    return pd.concat([min_values, max_values]).sort_values(['subject_id', 'test_name', 'time'])

def calculate_survival_time(patient_data, outcome_column):
    patient_data = patient_data.copy()
    patient_data[outcome_column] = pd.to_datetime(patient_data[outcome_column])
    patient_data['censorship_date'] = pd.to_datetime(patient_data['censorship_date'])
    patient_data['time'] = pd.to_datetime(patient_data['time'])
    
    patient_data['time_to_event'] = np.where(
        patient_data[outcome_column].isna(),
        (patient_data['censorship_date'] - patient_data['time']).dt.days,
        (patient_data[outcome_column] - patient_data['time']).dt.days
    )
    
    patient_data['event_status'] = np.where(patient_data[outcome_column].isna(), 0, 1)
    patient_data = patient_data[patient_data['time_to_event'] > 0]
    # if event status is 1, remove people who have a time_to_event < 10
    patient_data = patient_data[~((patient_data['event_status'] == 1) & (patient_data['time_to_event'] < 1))]
    
    #columns_to_drop = [outcome_column, 'censorship_date', 'time']
    #patient_data.drop(columns=columns_to_drop, inplace=True)
    
    return patient_data

def check_within_interval(data, value_column, mean_column, variance_column, interval_type):
    data = data.copy()
    std_dev = np.sqrt(data[variance_column])
    mean = data[mean_column]
    
    lower_bound = mean - 2*std_dev
    upper_bound = mean + 2*std_dev
    
    interval_name = f'not_within_{interval_type}_interval'
    data[interval_name] = np.where(
        (data[value_column] >= lower_bound) & (data[value_column] <= upper_bound),
        0, 1
    )
    
   # data.drop(columns=[mean_column, variance_column], inplace=True)
    return data

def convert_to_category(data, column_name):
    data[column_name] = data[column_name].astype('category').cat.codes
    return data

def validate_survival_data(data):
    assert (data['time_to_event'] > 0).all(), "Time to event is not positive"
    assert ((data['event_status'] == 0) | (data['event_status'] == 1)).all(), "Event status is not 0 or 1"

def stratified_train_test_split(data, test_size=0.2, random_state=42):
    try: 
        data['time_to_event_bin'] = pd.cut(data['time_to_event'],
                                        bins=np.arange(data['time_to_event'].min(), 
                                                    data['time_to_event'].max() + 365*2, 365*2),
                                        include_lowest=True, right=False)
        
        data['stratify_group'] = (data['test_name'].astype(str) + '_' + 
                                data['event_status'].astype(str) + '_' #+ 
                                #data['time_to_event_bin'].astype(str)
                                )
        
        train_data, test_data = train_test_split(
            data, test_size=test_size, stratify=data['stratify_group'], random_state=random_state
        )
        
        train_data = train_data.drop(['time_to_event_bin', 'stratify_group'], axis=1)
        test_data = test_data.drop(['time_to_event_bin', 'stratify_group'], axis=1)
    # otherwise, just split by the test_name
    except:
        train_data, test_data = train_test_split(
            data, test_size=test_size, stratify=data['test_name'], random_state=random_state
        )
        
    return train_data, test_data

def cox_regression(patient_group, outcome_name, interval):
    interval_col = f'not_within_{interval}_interval'
    model_variables[2] = interval_col
    cox_model = CoxPHFitter()
    cox_model.fit(patient_group[model_variables], 'time_to_event', 'event_status', formula=f"age + sex + not_within_{interval}_interval")
    
    # Save individual model
    model_filename = f'../results/models/{outcome_name}/cox_model_{test_code}_{interval}.pkl'
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    with open(model_filename, 'wb') as f:
        pickle.dump(cox_model, f)
    cox_model_results = {
                        'model_path': model_filename,
                        'test_name': test_code,
                        'interval_type': interval,
                        'hazard_ratio': cox_model.summary.loc[interval_col, 'exp(coef)'],
                        'ci_lower': cox_model.summary.loc[interval_col, 'exp(coef) lower 95%'],
                        'ci_upper': cox_model.summary.loc[interval_col, 'exp(coef) upper 95%'],
                        'p_value': cox_model.summary.loc[interval_col, 'p']
                    }
    return cox_model_results

def fit_rsf_model(df, interval_col, model_path, config):
    
    df[interval_col] = df[interval_col].astype(int)
    features = ['age', 'sex', interval_col]
    X = df[features]
    y = Surv.from_arrays(df['event_status'].astype(bool), df['time_to_event'])

    rsf = RandomSurvivalForest(n_estimators=100, **config)
    rsf.fit(X, y)

    with open(model_path, 'wb') as f:
        pickle.dump(rsf, f)

    return {
        'model_path': model_path,
        'test_name': test_code,
        'interval_type': interval,
        'hazard_ratio': None,
        'ci_lower': None,
        'ci_upper': None,
        'p_value': None
    }


def run_survival_analysis(data, outcome_name, interval_types, model_type='regression'):
    results = []
    model_variables = ['age', 'sex', None, 'time_to_event', 'event_status']
    
    for test_code, patient_group in data.groupby('test_name'):
        patient_group = patient_group.copy()
        patient_group['sex'] = patient_group['sex'].astype('category').cat.codes
        
        for interval in interval_types:
            try:
                if model_type == 'regression': 
                    interval_col = f'not_within_{interval}_interval'
                    model_variables[2] = interval_col
                    cox_model = CoxPHFitter()
                    cox_model.fit(patient_group[model_variables], 'time_to_event', 'event_status', formula=f"age + sex + not_within_{interval}_interval")
                    
                    if cox_model.summary.loc[interval_col, 'exp(coef) upper 95%'] < 100:
                        # Save individual model
                        model_filename = f'../results/models/{outcome_name}/cox_model_{test_code}_{interval}.pkl'
                        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
                        with open(model_filename, 'wb') as f:
                            pickle.dump(cox_model, f)
                        cox_model_results = {
                                            'model_path': model_filename,
                                            'test_name': test_code,
                                            'num_measurements': len(patient_group),
                                            'interval_type': interval,
                                            'hazard_ratio': cox_model.summary.loc[interval_col, 'exp(coef)'],
                                            'ci_lower': cox_model.summary.loc[interval_col, 'exp(coef) lower 95%'],
                                            'ci_upper': cox_model.summary.loc[interval_col, 'exp(coef) upper 95%'],
                                            'p_value': cox_model.summary.loc[interval_col, 'p']
                                        }
                        results.append(cox_model_results)
                else:
                    pass #fit_rsf_model(patient_group, interval, model_path, config)
                    
            except Exception as e:
                print(f"Error fitting model for {test_code} and {interval}: {e}")
                pass
    
    return pd.DataFrame(results)


def predict_survival(model_path, data, time_points):
    with open(model_path, 'rb') as f:
        cox_model = pickle.load(f)
    survival_probabilities = cox_model.predict_survival_function(data, times=[x*365 for x in time_points])
    risk_probabilities = 1 - survival_probabilities
    
    # Create DataFrame with risk probabilities and labels
    labels = [f'{int(x)}yr' for x in time_points]
    risk_df = pd.DataFrame(risk_probabilities.T)
    risk_df.columns = labels
    
    # Add event indicators - only count as event if they had the event within the time window
    # and were followed up for at least that long
    for i, time_point in enumerate(time_points):
        risk_df[f'event_{time_point}yr'] = np.where((data['event_status'] == 1) & (data['time_to_event'] <= time_point*365), 1, 0)

    return risk_df

def bootstrap_metric(y_true, y_pred, metric_fn, n_bootstraps=1000):
    # Check if there are enough unique values
        
    n_samples = len(y_true)
    bootstrap_values = []
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n_samples, n_samples)
        
        # Get bootstrap sample
        y_true_boot = y_true.reset_index(drop=True)[indices]
        y_pred_boot = y_pred.reset_index(drop=True)[indices]
        
        # Skip if bootstrap sample doesn't have enough unique values
        if len(np.unique(y_true_boot)) < 2 or len(np.unique(y_pred_boot)) < 2:
            continue
            
        value = metric_fn(y_true_boot, y_pred_boot)
        bootstrap_values.append(value)
    
    # Return None if no valid bootstrap samples
    if len(bootstrap_values) == 0:
        return {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None
        }
        
    return {
        'mean': np.mean(bootstrap_values),
        'ci_lower': np.percentile(bootstrap_values, 2.5),
        'ci_upper': np.percentile(bootstrap_values, 97.5)
    }

def calculate_auc(y_true, y_pred):
    """Calculate AUC score."""
    return roc_auc_score(y_true, y_pred)

def calculate_sensitivity(y_true, y_pred, threshold=0.5):
    """Calculate sensitivity (true positive rate)."""
    predicted_positive = y_pred >= threshold
    true_positive = (predicted_positive) & (y_true == 1)
    return true_positive.sum() / (y_true == 1).sum()

def calculate_specificity(y_true, y_pred, threshold=0.5):
    """Calculate specificity (true negative rate)."""
    predicted_positive = y_pred >= threshold
    true_negative = (~predicted_positive) & (y_true == 0)
    return true_negative.sum() / (y_true == 0).sum()

def plot_risk_probabilities(predictions):
    # Get columns containing risk probabilities (those with 'yr' but not 'event')
    time_points = [col for col in predictions.columns if 'yr' in col and 'event' not in col]
    
    # Create single figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot KDE for each time point with different colors
    for time_point in time_points:
        sns.kdeplot(data=predictions[time_point], label=time_point, ax=ax)
    
    ax.set_title('Risk Probability Distribution by Time Point')
    ax.set_xlabel('Risk Probability') 
    ax.set_ylabel('Density')
    ax.legend()
    
    plt.tight_layout()
    return fig


def main():
    # Load data
    reference_data = pd.read_csv('../bayes/results/bayes_reference_prior.csv')
    lab_measurements = pd.read_csv('../data/processed/lab_measurements.csv')[['subject_id', 'test_name', 'numeric_value', 'time', 'age']]
    
    # Create simulated outcomes
    outcomes = create_simulated_outcomes(reference_data, lab_measurements)

    print('Calculated Reference Intervals', reference_data.head(3))
    print('Raw Lab Measurements', lab_measurements.head(3))
    print('Simulated Outcomes', outcomes.head(3))
    
    processed_data = extract_followup_labs(reference_data, lab_measurements, min_days=10, max_days=365*2)
    print('Extracted Followup Labs', processed_data.head(3))
    
    processed_data = processed_data.merge(outcomes, on=['subject_id'], how='left')
    print('Merged Data', processed_data.head(3))
    
    processed_data = calculate_survival_time(processed_data, 'mortality_date')
    print('Calculated Survival Time', processed_data.head(3))
    
    processed_data = check_within_interval(processed_data, 'numeric_value', 'prior_mean', 'prior_var', 'reference')
    processed_data = check_within_interval(processed_data, 'numeric_value', 'obs_mean', 'obs_var', 'observed')
    processed_data = check_within_interval(processed_data, 'numeric_value', 'post_mean', 'post_var', 'bayes')
    print('Within Interval Data', processed_data.head(3))
        
    survival_results = run_survival_analysis(processed_data)
    print('Survival Results', survival_results.head(3))
    
    
    plot_hazard_ratios(survival_results, outcome_name='mortality')

if __name__ == "__main__":
    main()