import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from process.config import *

np.random.seed(42)

def calculate_subject_statistics(df):
    stats_list = []
    grouped = df.groupby(['subject_id', 'test_name'])
    
    for (subject_id, test_name), group in tqdm(grouped, desc="Calculating Subject Statistics"):
        group['time'] = pd.to_datetime(group['time'], errors='coerce')
        group = group.dropna(subset=['time'])
        
        if group.empty:
            continue
            
        sex = group['sex'].iloc[0] if 'sex' in group.columns else None
        age = max(group['age']) if 'age' in group.columns else None
        
        first_time = group['time'].min()
        last_time = group['time'].max()
        time_period_days = (last_time - first_time).days
        
        # Group statistics
        mean = group['numeric_value'].mean()
        var = group['numeric_value'].var()
        count = len(group)
        
        stats_dict = {
            'subject_id': subject_id,
            'test_name': test_name,
            'sex': sex,
            'age': age,
            'num_measurements': count,
            'last_time': last_time,
            'time_period_days': time_period_days,
            'obs_mean': mean,
            'obs_var': var
            
        }
        
        stats_list.append(stats_dict)
    
    stats_df = pd.DataFrame(stats_list)
    return stats_df.sort_values(by=['subject_id', 'test_name'])

def get_priors_dict():
    priors = {
    (test, sex): {
        'mean': (low + high) / 2,
        'var': ((high - low) / 4)**2
    }
    for test, intervals in REFERENCE_INTERVALS.items()
        for sex, (low, high, _) in intervals.items()
    }
    return priors
 
def get_reference_priors(row, priors):
    """Get reference interval based priors for a test/sex combination"""
    key = (row['test_name'], row['sex'])
    prior_row = priors[(key[0], key[1])]
    return prior_row['mean'], prior_row['var']

def get_empirical_priors(df, test_name, sex, priors):
    """Estimate priors empirically from data within reasonable bounds"""
    prior_row = priors[(test_name, sex)]
    ref_mean, ref_var = prior_row['mean'], prior_row['var']
    ref_sd = np.sqrt(ref_var)
    valid_data = df.loc[    
        (df['test_name'] == test_name) & 
        (df['sex'] == sex) &
        (df['obs_mean'].between(ref_mean - 2*ref_sd, ref_mean + 2*ref_sd))
    ]
    
    mu0 = np.mean(valid_data['obs_mean'])
    tau2 = np.mean(valid_data['obs_var'])
    return mu0, tau2