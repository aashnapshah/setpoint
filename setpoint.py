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
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing CBC data")
    parser.add_argument("--output_dir", type=str, default="data/setpoint_calculations", help="Output directory")
    return parser.parse_args()

def process_dataframe(df):
    """Process DataFrame with time and numeric_value columns."""
    x = df['time'].values.astype(np.datetime64)
    y = df['numeric_value'].values
    return x, y

def process_data(x, y):
    """Process raw time series data."""
    # Convert inputs to numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Sort chronologically
    sorted_idx = np.argsort(x.flatten())
    x = x[sorted_idx]
    y = y[sorted_idx]
    
    # Convert datetime to numeric days
    if isinstance(x[0], (np.datetime64, datetime)):
        x = (x - x[0]).astype('timedelta64[D]').astype(int)
    
    # Clean data
    valid_idx = ~np.isnan(y)
    x = x[valid_idx].reshape(-1, 1)
    y = y[valid_idx].reshape(-1, 1)
    
    # Remove duplicates
    unique_x, unique_idx = np.unique(x, return_index=True)
    return unique_x.reshape(-1, 1), y[unique_idx]

def filter_measurements(x, y, min_gap, min_tests):
    """Filter measurements by time gap and minimum tests."""
    if len(x) < min_tests:
        return None, None
        
    # Filter by minimum time gap
    nearest_marker = np.array([
        np.min(np.abs(x[i] - np.setdiff1d(x, x[i]))) 
        for i in range(len(x))
    ])
    valid_idx = nearest_marker > min_gap
    x = x[valid_idx]
    y = y[valid_idx]
    
    return (x, y) if len(x) >= min_tests else (None, None)

def fit_gmm(y):
    """Fit GMM models and select best fit."""
    y = y.reshape(-1, 1)
    
    # Common GMM parameters
    gmm_params = {
        'max_iter': 1000,
        'reg_covar': 1e-6,
        'random_state': 42,
        'init_params': 'kmeans'
    }
    
    # Try 2-component GMM
    gmm = GaussianMixture(n_components=2, **gmm_params).fit(y)
    
    # Check if 2-component model is good
    if gmm.converged_ and np.max(gmm.weights_) > 0.7:
        max_idx = np.argmax(gmm.weights_)
        return (
            gmm.means_[max_idx][0],
            np.sqrt(gmm.covariances_[max_idx][0][0]) / gmm.means_[max_idx][0],
            'gmm'
        )
    
    # Try 3-component GMM
    gmm = GaussianMixture(n_components=3, **gmm_params).fit(y)
    
    # Check if 3-component model is good
    if gmm.converged_ and np.max(gmm.weights_) > 0.45:
        max_idx = np.argmax(gmm.weights_)
        return (
            gmm.means_[max_idx][0],
            np.sqrt(gmm.covariances_[max_idx][0][0]) / gmm.means_[max_idx][0],
            'gmm'
        )
    
    # Fall back to basic statistics
    return np.mean(y), np.std(y) / np.mean(y), 'statistical'

def calculate_setpoint(x, y, min_gap, min_tests):
    """Calculate physiological setpoint from input data."""
    x, y = process_data(x, y)
    x, y = filter_measurements(x, y, min_gap, min_tests)
    if x is None:
        return np.nan, np.nan, "insufficient_data"
    
    return fit_gmm(y)


def main():
    args = parse_args()
    
    df = pd.read_csv(os.path.join(args.data_dir, 'subject_cbc_events.csv'))

    # Process each group
    setpoints = []
    insufficient_data = []
    
    for (subject, code), group in df.groupby(['subject_id', 'code']):
        if not group.empty:
            x, y = process_dataframe(group)
            setpoint, uncertainty, model = calculate_setpoint(x, y, args.min_gap, args.min_tests)
            
            if model == "insufficient_data":
                insufficient_data.append({
                    'subject_id': subject,
                    'code': code,
                    'n_measurements': len(group)
                })
                print(f"Insufficient data for subject {subject}, code {code} ({len(group)} measurements)")
            else:
                setpoints.append({
                    'subject_id': subject,
                    'code': code,
                    'setpoint': setpoint,
                    'uncertainty': uncertainty,
                    'model_type': model
                })

    # Save results
    if setpoints:
        setpoint_df = pd.DataFrame(setpoints)
        output_file = os.path.join(args.output_dir, f'setpoints_gap:{args.min_gap}_tests:{args.min_tests}.csv')
        setpoint_df.to_csv(output_file, index=False)
        print(f"\nFound {len(setpoints)} valid setpoints")
        print(f"Results saved to {output_file}")
    
    # Analyze insufficient data cases
    if insufficient_data:
        insuff_df = pd.DataFrame(insufficient_data)
        print("\nInsufficient data summary by code:")
        summary = insuff_df.groupby('code').agg({
            'subject_id': 'count',
            'n_measurements': ['mean', 'min', 'max']
        })
        summary.columns = ['n_subjects', 'avg_measurements', 'min_measurements', 'max_measurements']
        print(summary)
        
        # Save insufficient data cases
        insuff_file = os.path.join(args.output_dir, f'insufficient_gap:{args.min_gap}_tests:{args.min_tests}.csv')
        insuff_df.to_csv(insuff_file, index=False)
        print(f"\nInsufficient data cases saved to {insuff_file}")

if __name__ == "__main__":
    main()