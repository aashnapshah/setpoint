# Code to calculate setpoints for EHRShot Data with set min. # of measurements and year cutoff

import pandas as pd
import numpy as np
import pickle as pkl
import sklearn as skl
from tqdm import tqdm
import sys
sys.path.append('../')
from process.config import *

# helper for calculate_setpoint
def calculate_gaussian_aic(measurements):
    num_points = len(measurements)
    mean = np.mean(measurements) 
    variance = np.var(measurements)  # MLE: ddof=0
    log_likelihood = -0.5 * num_points * (np.log(2 * np.pi * variance) + 1)
    num_parameters = 2
    aic_score = 2 * num_parameters - 2 * log_likelihood

    mean = np.mean(measurements)
    std_dev = np.std(measurements, ddof=0)
    coeff_variation = std_dev / mean
    variance = std_dev**2

    return aic_score, mean, std_dev, coeff_variation, variance

# basic fxn to calculate setpoints
def calculate_gaussians_thresh(measurements: np.ndarray, max_gaussian_components=3, mixture_thresholds={2:0.7, 3:0.45}):
    # for 1 component model it is faster to use analytic expr for aic
    best_aic_score = -1
    best_aic_score, mean, std_dev, coeff_variation, variance = calculate_gaussian_aic(measurements)
    best_mixture_model = None

    # fit 2/3 component models
    for num_components in range(2, max_gaussian_components + 1):
        gaussian_mixture = skl.mixture.GaussianMixture(
            n_components=num_components,
            covariance_type='full',
            max_iter=300,
            reg_covar=0.001,
            random_state=0
        ).fit(measurements.reshape(-1, 1))

        aic_score = gaussian_mixture.aic(measurements.reshape(-1, 1))
        if aic_score < best_aic_score and gaussian_mixture.converged_:
            best_aic_score = aic_score
            best_mixture_model = gaussian_mixture

    # Check if 2/3 component model beat 1 component aic
    if best_mixture_model is not None:
        component_weights = best_mixture_model.weights_
        dominant_component_idx = np.argmax(component_weights)
        dominant_weight = component_weights[dominant_component_idx]
        dominance_threshold = mixture_thresholds[best_mixture_model.n_components]

        # check dominant component has sufficient size
        if dominant_weight > dominance_threshold:
            mean = best_mixture_model.means_[dominant_component_idx, 0]
            covariance = best_mixture_model.covariances_[dominant_component_idx, 0, 0]
            std_dev = np.sqrt(covariance)
            coeff_variation = std_dev / mean
            variance = std_dev**2
            return mean, std_dev, coeff_variation, variance

    # fallback to 1-component model statistics 
    return mean, std_dev, coeff_variation, variance

def calculate_gaussians_health(measurements: np.ndarray, healthy: float, max_gaussian_components=3):
    measurements_reshaped = measurements.reshape(-1, 1)

    # Start with 1-component model
    best_aic_score = -1
    best_model = None
    best_aic_score, mean, std_dev, coeff_variation, variance = calculate_gaussian_aic(measurements)

    # Try mixture models
    for num_components in range(2, max_gaussian_components + 1):
        gm = skl.mixture.GaussianMixture(
            n_components=num_components,
            covariance_type='full',
            max_iter=300,
            reg_covar=0.001,
            random_state=0
        ).fit(measurements_reshaped)

        aic_score = gm.aic(measurements_reshaped)
        if aic_score < best_aic_score and gm.converged_:
            best_aic_score = aic_score
            best_model = gm

    # Use best model's component closest to `healthy` parameter
    if best_model != None:
        means = best_model.means_.flatten()
        covariances = best_model.covariances_.reshape(-1)  # 1D array of variances
        closest_idx = np.argmin(np.abs(means - healthy))

        mean = means[closest_idx]
        variance = covariances[closest_idx]
        std_dev = np.sqrt(variance)
        coeff_variation = std_dev / mean

        return mean, std_dev, coeff_variation, variance
    

    return mean, std_dev, coeff_variation, variance

def run_gmm_model(measurements_df, cluster_picker='healthy'):
    setpoint_records = []
    patient_test_groups = measurements_df.groupby(['subject_id', 'test_name'])

    for (patient_id, test_name), measurement_group in tqdm(patient_test_groups, desc="Calculating Setpoints"):
        sex = measurement_group['sex'].iloc[0]
        low, high = REFERENCE_INTERVALS[test_name][sex][0], REFERENCE_INTERVALS[test_name][sex][1]
        med = (low+high)/2

        if cluster_picker == 'healthy':
            setpoint_mean, setpoint_std, setpoint_cv, setpoint_var = calculate_gaussians_health(
                measurement_group['numeric_value'].values, 
                healthy=med
            )
        elif cluster_picker == 'threshold':
            setpoint_mean, setpoint_std, setpoint_cv, setpoint_var = calculate_gaussians_thresh(
                measurement_group['numeric_value'].values
            )
        else:
            raise Exception("Not a valid cluster_picker argument, must be 'healthy' or 'thresh'")

        latest_measurement_date = measurement_group['time'].max()
        age = max(measurement_group['age']) if 'age' in measurement_group.columns else None

        setpoint_record = {
            'subject_id': patient_id,
            'test_name': test_name,
            'sex':sex,
            'age':age,
            'setpoint_mean': setpoint_mean,
            'setpoint_var': setpoint_var,
            'setpoint_date': latest_measurement_date,
            'prior_mean': None,
            'priot_var': None,
        }
        setpoint_records.append(setpoint_record)

    setpoints_df = pd.DataFrame(setpoint_records)
    
    return setpoints_df
    
if __name__ == "__main__":
    measurements_df = pd.read_pickle("data/filter/labs_filter_foy_30_5_2018.pkl")
    setpoints_df = run_gmm_model(measurements_df)
    setpoints_df.to_pickle("data/setpoints_df_2018.pkl")