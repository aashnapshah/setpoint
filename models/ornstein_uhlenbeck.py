import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm

# append path
sys.path.append('../')
from process.tables import *
from models.bayes import *
from models.predict import *
from process.config import *

def log_likelihood_ou_with_prior(parameters, S, dt, prior_mean, prior_var, prior_logsigma_mean, prior_logsigma_var):
    """OU log-likelihood with proper MAP prior"""
    theta, mu, sigma = parameters

    if mu <= 0 or sigma <= 0:
        return -np.inf

    N = len(S)
    log_likelihood = 0

    for i in range(1, N):
        x_prev = S[i - 1]
        x_curr = S[i]

        # OU transition parameters
        exp_term = np.exp(-mu * dt)
        expected = x_prev * exp_term + theta * (1 - exp_term)
        variance = (sigma**2 / (2 * mu)) * (1 - np.exp(-2 * mu * dt))

        if variance <= 0 or np.isnan(variance):
            return -np.inf

        log_likelihood += -0.5 * np.log(2 * np.pi * variance) - 0.5 * ((x_curr - expected) ** 2) / variance

    # Prior on theta ~ N(prior_mean, prior_var)
    prior_ll_theta = -0.5 * np.log(2 * np.pi * prior_var) - 0.5 * ((theta - prior_mean) ** 2) / prior_var

    # Prior on log(sigma) ~ N(logsigma_mean, logsigma_var)
    log_sigma = np.log(sigma)
    prior_ll_logsigma = -0.5 * np.log(2 * np.pi * prior_logsigma_var) - \
                        0.5 * ((log_sigma - prior_logsigma_mean) ** 2) / prior_logsigma_var \
                        - log_sigma  # Jacobian adjustment for log transform

    return log_likelihood + prior_ll_theta + prior_ll_logsigma

def estimate_ou_parameters_with_prior(S, dt, prior_mean, prior_var, prior_sigma_mean, prior_sigma_var):
    """Estimate OU parameters using MAP with proper priors"""

    # Convert to log-scale for sigma prior
    prior_logsigma_mean = np.log(prior_sigma_mean)
    prior_logsigma_var = prior_sigma_var

    theta0 = prior_mean
    mu0 = 1/30 # Start with small mean reversion speed
    sigma0 = prior_sigma_mean

    initial_guess = [theta0, mu0, sigma0]
    result = minimize(
        fun=lambda params: -log_likelihood_ou_with_prior(params, S, dt,
                                                         prior_mean, prior_var,
                                                         prior_logsigma_mean, prior_logsigma_var),
        x0=initial_guess,
        bounds=[(None, None), (1/365, 1), (1e-3, sigma0)],  # Bound μ Bound μ (1/10 is 10 days) and sigma0 is 1000
        method='L-BFGS-B'
    )

    return result.x if result.success else initial_guess

def run_ou_with_prior(processed_df):
    """Run OU estimation with proper priors"""
    groups = processed_df.groupby(['subject_id', 'test_name'])
    priors = get_priors_dict()
    results = []
    
    for (subject, test_name), group in tqdm(groups, desc="Fitting OU processes"):
        try:
            sample = group[['time', 'numeric_value', 'sex']]
            sex = sample['sex'].unique()[0]
            prior_mean, prior_var = priors[(test_name, sex)]['mean'], priors[(test_name, sex)]['var'] 
            prior_sigma_mean = np.sqrt(prior_var) 
            prior_sigma_var = 0.2   # log-variance of sigma (log-normal prior)
            
            filtered_sample = sample[sample['numeric_value'].between(prior_mean - 2 * prior_sigma_mean, prior_mean + 2 * prior_sigma_mean)]     
            df = filtered_sample.copy()
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')
            df['dt'] = df['time'].diff().dt.total_seconds().div(86400)
            df = df.dropna()
            
            S = df['numeric_value'].values
            dt = df['dt'].mean()
            
            if len(df) < 2:  # Need at least 2 points for OU
                results.append({
                    'subject_id': subject,
                    'test_name': test_name,
                    'sex': sex,
                    'ou_prior_mean': prior_mean,
                    'ou_prior_var': prior_var,
                    'ou_prior_sigma_mean': prior_sigma_mean,
                    'ou_prior_sigma_var': prior_sigma_var,
                    'ou_mean': prior_mean,
                    'ou_speed': np.nan,
                    'ou_std': prior_sigma_mean,
                    'ou_var': prior_sigma_mean**2,
                    'n_measurements': 0,
                    'dt_mean': 0
                })
                continue
            
            theta_ml, mu_ml, sigma_ml = estimate_ou_parameters_with_prior(
                S, dt, prior_mean, prior_var, prior_sigma_mean, prior_sigma_var
            )
            ou_std_dev = min(prior_sigma_mean, sigma_ml / np.sqrt(2 * mu_ml))
            
            results.append({
                'subject_id': subject,
                'test_name': test_name,
                'sex': sex,
                'ou_prior_mean': prior_mean,
                'ou_prior_var': prior_var,
                'ou_prior_sigma_mean': prior_sigma_mean,
                'ou_prior_sigma_var': prior_sigma_var,
                'ou_mean': theta_ml,
                'ou_speed': mu_ml,
                'ou_std': ou_std_dev,
                'ou_var': ou_std_dev**2,
                'n_measurements': len(S),
                'dt_mean': dt
            })
            
        except Exception as e:
            print(f"Error processing {subject}, {test_name}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    return results_df

# Usage example
if __name__ == "__main__":
    # Load your processed data
    # processed_df = your_data_loading_function()
    
    # Run OU estimation
    ou_results = run_ou_with_prior(processed_df)
    
    # Check results
    print("OU Results Summary:")
    print(ou_results.describe())
    
    # Check for reasonable parameter ranges
    print("\nParameter Ranges:")
    print(f"θ (mean reversion level): {ou_results['ou_mean'].min():.2f} - {ou_results['ou_mean'].max():.2f}")
    print(f"μ (mean reversion speed): {ou_results['ou_speed'].min():.4f} - {ou_results['ou_speed'].max():.4f}")
    print(f"σ (volatility): {ou_results['ou_std'].min():.2f} - {ou_results['ou_std'].max():.2f}")