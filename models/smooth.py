import pandas as pd     
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import norm  
from statsmodels.nonparametric.smoothers_lowess import lowess  

import pymc as pm   
import arviz as az      
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def centered_moving_average(timeseries, window_days=60):
    """Centered moving average with time-based window."""
    return timeseries.rolling(window=f"{window_days}D", center=True, min_periods=1).mean()

def gaussian_kernel_smooth(timeseries, window_days=60, std_days=15.0):
    """Custom time-aware Gaussian smoothing."""
    smoothed_values = []
    for timestamp in timeseries.index:
        window_start = timestamp - pd.Timedelta(days=window_days / 2)
        window_end = timestamp + pd.Timedelta(days=window_days / 2)
        window_data = timeseries[(timeseries.index >= window_start) & (timeseries.index <= window_end)]

        if len(window_data) == 0:
            smoothed_values.append(np.nan)
            continue

        time_deltas = (window_data.index - timestamp).days.astype(float)
        kernel_weights = norm.pdf(time_deltas, loc=0, scale=std_days)
        kernel_weights /= kernel_weights.sum()

        smoothed_value = np.dot(window_data.values.flatten(), kernel_weights)
        smoothed_values.append(smoothed_value)

    return pd.Series(smoothed_values, index=timeseries.index)

def exponential_weighted_smooth(timeseries, span=10):
    """Exponential weighted moving average."""
    return timeseries.ewm(span=span, adjust=False).mean()

def lowess_smooth(timeseries, frac=0.7):
    """LOWESS smoothing (locally weighted scatterplot smoothing)."""
    smoothed_values = lowess(timeseries.values.flatten(), timeseries.index.values.astype(float), frac=frac, return_sorted=False)
    
    # Calculate mean
    mean_value = np.mean(smoothed_values)
    
    # Calculate 95% confidence interval
    ci_95 = 1.96 * np.std(smoothed_values)
    lower_ci = mean_value - ci_95
    upper_ci = mean_value + ci_95
    
    # Calculate mean of the confidence interval
    ci_mean = (lower_ci + upper_ci) / 2
    
    # Get the max, min, and mean of the smoothed values
    max_value = np.max(smoothed_values)
    min_value = np.min(smoothed_values)
    std_x_value = (max_value - min_value) / 4   
    mean_value = (max_value + min_value) / 2
    
    return smoothed_values, ci_mean, np.std(smoothed_values), mean_value, std_x_value

def plot_series_with_toggle(axis, series_dict):
    """Plot all series and make legend items toggleable."""
    plot_lines = []
    for label, series in series_dict.items():
        line, = axis.plot(series, label=label)
        plot_lines.append(line)

    legend = axis.legend(loc="upper left")
    for legend_line in legend.get_lines():
        legend_line.set_picker(5)

    line_map = dict(zip(legend.get_lines(), plot_lines))

    def on_pick(event):
        legend_line = event.artist
        original_line = line_map[legend_line]
        visibility = not original_line.get_visible()
        original_line.set_visible(visibility)
        legend_line.set_alpha(1.0 if visibility else 0.2)
        axis.figure.canvas.draw()

    axis.figure.canvas.mpl_connect('pick_event', on_pick)


def bayesian_gp_smooth(timeseries, num_points=200):
    """Bayesian Gaussian Process smoothing using PyMC."""
    days_elapsed = (timeseries.index - timeseries.index[0]).days.values.astype(float)
    measurements = timeseries.values.flatten()

    prediction_days = np.linspace(days_elapsed.min(), days_elapsed.max(), num_points)

    with pm.Model() as model:
        # Priors
        amplitude = pm.HalfNormal("amplitude", sigma=1.0)
        length_scale = pm.HalfNormal("length_scale", sigma=10.0)
        noise = pm.HalfNormal("noise", sigma=1.0)

        # Covariance kernel
        covariance = amplitude**2 * pm.gp.cov.ExpQuad(1, length_scale)

        gp = pm.gp.Marginal(cov_func=covariance)

        observations = gp.marginal_likelihood("observations", X=days_elapsed[:, None], y=measurements, noise=noise)
        
        # Prediction
        predictions = gp.conditional("predictions", Xnew=prediction_days[:, None])
        trace = pm.sample(1000, chains=2, tune=1000, target_accept=0.95, return_inferencedata=True, progressbar=False)

    posterior_mean = trace.posterior["predictions"].mean(("chain", "draw")).values
    posterior_std = trace.posterior["predictions"].std(("chain", "draw")).values
    smoothed_series = pd.Series(posterior_mean, index=pd.to_datetime(timeseries.index[0]) + pd.to_timedelta(prediction_days, unit='D'))
    return smoothed_series, posterior_std, prediction_days


def sklearn_gpr_smooth(timeseries: pd.Series, num_points=200, length_scale=30.0):
    """Gaussian Process Regression smoothing using scikit-learn (no sampling)."""
    # Convert datetime index to numeric (days since start)
    days_elapsed = (timeseries.index - timeseries.index[0]).days.values.reshape(-1, 1)
    measurements = timeseries.values

    # Prediction grid
    prediction_days = np.linspace(days_elapsed.min(), days_elapsed.max(), num_points).reshape(-1, 1)
    prediction_dates = timeseries.index[0] + pd.to_timedelta(prediction_days.flatten(), unit='D')

    # Kernel: signal variance × RBF kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=True)

    # Fit & predict
    gaussian_process.fit(days_elapsed, measurements)
    predicted_mean, predicted_std = gaussian_process.predict(prediction_days, return_std=True)

    # Return smoothed mean series and std
    return pd.Series(predicted_mean, index=prediction_dates), pd.Series(predicted_std, index=prediction_dates)

def run_lowess_model(df, smooth_frac=0.70):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    results = []

    for (subject_id, test_name), group in df.groupby(['subject_id', 'test_name']):
        seq = group[['time', 'numeric_value']].drop_duplicates().set_index('time').sort_index()
        curve, mean, std, x_mean, x_std = lowess_smooth(seq, frac=smooth_frac)
        results.append({
            'subject_id': subject_id,
            'test_name': test_name,
            'smoothed_curve': curve,
            'smoothed_mean': mean,
            'smoothed_var': std**2, 
            'smoothed_x_var': x_std**2,
            'smoothed_x_mean': x_mean
        })
    return pd.DataFrame(results)


def main(subject_id = 115967141, test_name = 'HGB'):
    measurements_df = pd.read_pickle("data/preprocess_df_no_freq_filter.pkl")
    patient_timeseries = measurements_df[(measurements_df['subject_id'] == subject_id) & (measurements_df['test_name'] == test_name)][['time', 'numeric_value']]
    patient_timeseries = patient_timeseries.drop_duplicates().set_index('time').sort_index()

    # Smoothing methods
    smoothed_series = {
        "Original": patient_timeseries,
        "Moving Avg (60D)": centered_moving_average(patient_timeseries, window_days=60),
        "Gaussian (std=15)": gaussian_kernel_smooth(patient_timeseries, window_days=60, std_days=15),
        "Exponential (span=10)": exponential_weighted_smooth(patient_timeseries, span=10),
        "LOWESS (frac=0.25)": lowess_smooth(patient_timeseries, frac=0.25),
    }

    # Plotting
    fig, axis = plt.subplots(figsize=(10, 6))
    plot_series_with_toggle(axis, smoothed_series)
    axis.set_title("Smoothing Methods on HGB Time Series")
    axis.set_xlabel("Time")
    axis.set_ylabel("HGB")
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    main(115967914, 'HGB')