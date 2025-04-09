import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from datetime import datetime
import os
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate physiological setpoints from CBC data")
    parser.add_argument("--min_gap", type=int, default=30, help="Minimum days between measurements")
    parser.add_argument("--min_tests", type=int, default=5, help="Minimum number of tests required")
    parser.add_argument("--year_cutoff", type=int, default='2016', help="Only include data from before this year")
    parser.add_argument("--data_dir", type=str, default="../data", help="Directory containing CBC data")
    parser.add_argument("--output_dir", type=str, default="results/setpoint_calculations", help="Output directory")
    return parser.parse_args()

def process_dataframe(df):
    """Process DataFrame with time and numeric_value columns."""
    x = df['time'].values.astype(np.datetime64)
    y = df['numeric_value'].values
    x, y, time = process_data(x, y)
    return x, y, time

def process_data(x, y):
    """Process raw time series data."""
    # Convert inputs to numpy arrays and sort chronologically
    x = np.array(x)
    y = np.array(y)
    time = x.copy()  
    
    sorted_idx = np.argsort(x.flatten())
    x = x[sorted_idx]
    y = y[sorted_idx]
    time = time[sorted_idx]
    
    # Convert datetime to numeric days
    if isinstance(x[0], (np.datetime64, datetime)):
        x = (x - x[0]).astype('timedelta64[D]').astype(int)
    
    # Clean data and remove NaN values
    valid_idx = ~np.isnan(y)
    x = x[valid_idx].reshape(-1, 1)
    y = y[valid_idx].reshape(-1, 1)
    time = time[valid_idx]
    
    # Remove duplicates
    unique_x, unique_idx = np.unique(x, return_index=True)
    return unique_x.reshape(-1, 1), y[unique_idx], time[unique_idx] 

def filter_measurements(x, y, time, min_gap, min_tests, year_cutoff):
    """Filter measurements by time gap and minimum tests."""
    
    year_cutoff = np.datetime64(f'{year_cutoff}-01-01')
    valid_idx = time < year_cutoff
    x = x[valid_idx]
    y = y[valid_idx]
    time = time[valid_idx]

    if len(x) < min_tests:
        return None, None, None
        
    # Filter by minimum time gap
    nearest_marker = np.array([
        np.min(np.abs(x[i] - np.setdiff1d(x, x[i]))) 
        for i in range(len(x))
    ])
    valid_idx = nearest_marker > min_gap
    
    if sum(valid_idx) < min_tests:
        return None, None, None
        
    return x[valid_idx], y[valid_idx], time[valid_idx]

def calculate_setpoint(y):
    """
    Fit GMM models and select best fit using AIC.
    
    1. Fits 1-3 Gaussian components
    2. Selects best model using AIC
    3. Returns mean of largest component as setpoint
    """
    y = y.reshape(-1, 1)
    
    # Common GMM parameters
    gmm_params = {
        'max_iter': 1000,
        'reg_covar': 1e-6,
        'random_state': 42,
        'init_params': 'kmeans'
    }
    
    # Check if we have enough unique values
    unique_values = np.unique(y)
    if len(unique_values) < 3:
        return np.mean(y), calculate_cv(y), np.std(y), 'statistical'
    
    try:
        # Fit models with 1-3 components
        models = []
        aics = []
        
        for n_components in range(1, 4):
            gmm = GaussianMixture(n_components=n_components, **gmm_params)
            gmm.fit(y)
            
            if gmm.converged_:
                # Calculate AIC: -2 * log-likelihood + 2 * n_parameters
                # For GMM, n_parameters = n_components * (mean + variance) + (n_components - 1) weights
                n_parameters = n_components * 2 + (n_components - 1)
                aic = -2 * gmm.score(y) * len(y) + 2 * n_parameters
                
                models.append(gmm)
                aics.append(aic)
            else:
                models.append(None)
                aics.append(np.inf)
        
        # Select best model (lowest AIC)
        if len(models) > 0 and min(aics) != np.inf:
            best_model_idx = np.argmin(aics)
            best_model = models[best_model_idx]
            
            # Get the largest component
            max_idx = np.argmax(best_model.weights_)
            
            return (
                best_model.means_[max_idx][0],
                calculate_cv(y[best_model.predict(y) == max_idx]),
                np.std(y[best_model.predict(y) == max_idx]),
                f'gmm_{best_model_idx + 1}_components'
            )
            
    except Exception as e:
        print(f"Warning: GMM fitting failed with error: {str(e)}. Falling back to statistical method.")
    
    # Fall back to basic statistics if GMM fails
    return np.mean(y), calculate_cv(y), 'statistical'

def calculate_cv(data):
    mean = np.mean(data)
    std = np.std(data)
    cv = (std / mean) * 100  # Convert to percentage
    return cv

def main():
    args = parse_args()
    
    df = pd.read_csv(os.path.join(args.data_dir, 'processed/cbc_measurements.csv'))
    
    # Process each group
    setpoints = []    
    for (subject, code), group in df.groupby(['subject_id', 'code']):
        if group.empty:
            continue
            
        # Process and filter measurements
        x, y, time = process_dataframe(group)
        x, y, time = filter_measurements(x, y, time, args.min_gap, args.min_tests, args.year_cutoff)
        if x is None:
            continue

        setpoint, cv, std, model = calculate_setpoint(y)
        setpoints.append({
            'subject_id': subject,
            'code': code,
            'setpoint': setpoint,
            'cv': cv,
            'std': std,
            'model_type': model, 
            'year_cutoff': args.year_cutoff,
            'min_gap': args.min_gap,
            'min_tests': args.min_tests,
            'setpoint_estimation_time': time[-1],
            'number_of_measurements': len(x), 
            'time_period': time[-1] - time[0]
        })

    # Save results
    if setpoints:
        setpoint_df = pd.DataFrame(setpoints)
        output_file = os.path.join(args.output_dir, f'setpoints_gap:{args.min_gap}_tests:{args.min_tests}_year:{args.year_cutoff}.csv')
        setpoint_df.to_csv(output_file, index=False, float_format='%.10g')

        print(f"Original number of subjects: {len(df.subject_id.unique())}")
        # For each code, print the number of subjects with valid setpoints
        for code in setpoint_df.code.unique():
            print(f"Number of subjects with valid setpoints for {code}: {len(setpoint_df[setpoint_df.code == code].subject_id.unique())}")
        print(f"\nFound valid setpoints for {len(setpoints)} subjects")
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()