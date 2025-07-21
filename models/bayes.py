import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from process.config import *

np.random.seed(42)


def calculate_subject_statistics(df, index_year=2018):
    stats_list = []
    grouped = df.groupby(['subject_id', 'test_name'])
    
    for (subject_id, test_name), group in tqdm(grouped, desc="Calculating Subject Statistics"):
        group['time'] = pd.to_datetime(group['time'], errors='coerce')
        group = group.dropna(subset=['time'])
        
        if group.empty:
            continue
            
        sex = group['sex'].iloc[0] if 'sex' in group.columns else None
        age = group['age'].iloc[-1] if 'age' in group.columns else None
        
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

def run_pEB_model(df, prior_method='reference'):
    df = df.copy()
    df = calculate_subject_statistics(df)
    priors = get_priors_dict()
    
    if prior_method == 'reference':
        df['prior_mean'], df['prior_var'] = zip(*df.apply(get_reference_priors, axis=1, priors=priors))
    
    elif prior_method == 'empirical':
        for test_name in df['test_name'].unique():
            for sex in df['sex'].unique():
                mask = (df['test_name'] == test_name) & (df['sex'] == sex)
                if mask.any():
                    mu0, tau2 = get_empirical_priors(df, test_name, sex, priors)
                    df.loc[mask, 'prior_mean'] = mu0
                    df.loc[mask, 'prior_var'] = tau2
    
    # Calculate posterior parameters
    df['w'] = df['prior_var'] / (df['prior_var'] + (df['obs_var']/df['num_measurements']))
    df['post_mean'] = df['w'] * df['obs_mean'] + (1 - df['w']) * df['prior_mean']
    df['post_var'] = (1/df['prior_var'] + df['num_measurements']/df['obs_var'])**(-1)
    return df

def run_hierarchical_bayes(obs_mean, obs_var, n_obs, max_iter=100, tol=1e-6):

    obs_mean = np.asarray(obs_mean)
    obs_var = np.asarray(obs_var)
    n_obs = np.asarray(n_obs)
    # Initialize
    mu0 = np.mean(obs_mean)
    tau2 = np.var(obs_mean)
    for _ in range(max_iter):
        se2 = obs_var / n_obs
        w = tau2 / (tau2 + se2)
        post_mean = w * obs_mean + (1 - w) * mu0
        post_var = (tau2 * se2) / (tau2 + se2)
        mu0_new = np.mean(post_mean)
        tau2_new = np.mean(post_var + (post_mean - mu0_new) ** 2)
        if np.abs(mu0_new - mu0) < tol and np.abs(tau2_new - tau2) < tol:
            break
        mu0, tau2 = mu0_new, tau2_new
    return mu0, tau2, post_mean, post_var

def plot_empirical_bayes(df, test_name, n_subjects=10):
    # Create separate plots for males and females
    # sample the same 20 patients for each sex every time
    df = df.copy().query('test_name == @test_name')
    sampled_df = df.groupby('sex').apply(lambda x: x.sample(n_subjects, random_state=42)).reset_index(drop=True)
    fig, (ax_m, ax_f) = plt.subplots(1, 2, figsize=(16, 5))
    np.random.seed(42)
    
    for sex, ax in [('M', ax_m), ('F', ax_f)]:
        # Filter data by sex
        sex_df = sampled_df[sampled_df['sex'] == sex] 
        if len(sex_df) == 0:
            continue

        # Y-ticks and labels
        sex_df['y_title'] = sex_df['subject_id'].astype(str) + ' N=' + sex_df['num_measurements'].astype(str)
        y_pos = np.arange(len(sex_df)) * 1.0  # space subjects
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sex_df['y_title'])

        # Palette
        colors = sns.color_palette("husl", 4)
        obs_color, post_color, emp_color, hier_color = colors

        # Standard deviations
        obs_std = np.sqrt(sex_df['obs_var'] / sex_df['num_measurements'])
        post_std = np.sqrt(sex_df['post_var'])
        prior_std = np.sqrt(sex_df['prior_var'].iloc[0])  # Use first value as they should all be same
        prior_mean = sex_df['prior_mean'].iloc[0]  # Use first value as they should all be same
        offset = 0.15

        # Plot layers
        ax.errorbar(sex_df['obs_mean'], y_pos - offset, xerr=2*obs_std, fmt='o', label='Observed (2σ)',
                    color=obs_color, alpha=0.8, capsize=2)

        ax.errorbar(sex_df['post_mean'], y_pos + offset, xerr=2*post_std, fmt='s', label='Posterior (Ref Prior)',
                    color=post_color, alpha=0.8, capsize=2)

        ax.axvline(prior_mean, color='gray', linestyle='--', label=f'Prior mean: {prior_mean:.2f}', alpha=0.6)
        ax.axvspan(prior_mean - 2*prior_std, prior_mean + 2*prior_std, alpha=0.1, color='gray', label='Prior (2σ)')

        if 'empirical_post_mean' in sex_df.columns:
            emp_std = np.sqrt(sex_df['empirical_post_var'])
            ax.errorbar(sex_df['empirical_post_mean'], y_pos + offset*2, xerr=2*emp_std, fmt='s', label='Empirical Posterior',
                        color=emp_color, alpha=0.8, capsize=2)

        if 'hierarchical_post_mean' in sex_df.columns:
            hier_std = np.sqrt(sex_df['hierarchical_post_var'])
            ax.errorbar(sex_df['hierarchical_post_mean'], y_pos + offset*3, xerr=2*hier_std, fmt='s', label='Hierarchical Posterior',
                        color=hier_color, alpha=0.8, capsize=2)

        # Labels & Theme
        ax.set_xlabel(f'{test_name} Value')
        ax.set_ylabel('Subject')
        ax.set_title(f'{test_name} - {sex}', fontsize=14)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='best', fontsize=9, frameon=False)
        sns.despine()

    plt.suptitle(f'Empirical Bayes for {test_name} by Sex', fontsize=16)
    plt.tight_layout()

    return fig, (ax_m, ax_f)
