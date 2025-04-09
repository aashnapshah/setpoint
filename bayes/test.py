import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import pandas as pd
from scipy import stats
import sys
import warnings
from sklearn.mixture import GaussianMixture

sys.path.append('../setpoint/')
from setpoint.setpoint import process_dataframe, process_data, filter_measurements, calculate_setpoint, calculate_cv, main

sys.path.append('../data_process/')
from processors.cbc import CBC_REFERENCE_INTERVALS

# Suppress warnings
warnings.filterwarnings('ignore')

def calculate_range(df, mean, std):
    """Calculate setpoint intervals."""
    coefficient_of_variation = std / mean if isinstance(mean, np.ndarray) else df[std] / df[mean]
    lower_bound = mean - 2 * coefficient_of_variation
    upper_bound = mean + 2 * coefficient_of_variation
    return lower_bound, upper_bound

def cluster_measurements(x, y, time):
    """Cluster CBC measurements for each subject using setpoint method."""
    # flatten x and y
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    time = time.reshape(-1, 1)

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
        return np.mean(y), calculate_cv(y), np.std(y), 'statistical', y

    try:
        # Fit models with 1-3 components
        models = []
        aics = []

        for n_components in range(1, 4):
            gmm = GaussianMixture(n_components=n_components, **gmm_params)
            gmm.fit(y)

            if gmm.converged_:
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

            # Predict cluster membership
            cluster_assignments = best_model.predict(y)
            max_idx = np.argmax(best_model.weights_)

            # Get samples belonging to the largest cluster
            idx = np.where(cluster_assignments == max_idx)
            cluster_x = x[idx]
            cluster_y = y[idx]
            cluster_time = time[idx]

            return cluster_x, cluster_y, cluster_time

    except Exception as e:
        print(f"Warning: GMM fitting failed with error: {str(e)}. Falling back to statistical method.")

    # Fall back to basic statistics if GMM fails
    return x, y, time

    
def process_measurements(df, code, min_gap, min_tests, year_cutoff):
    """Filter CBC measurements by code."""
    filtered_df = pd.DataFrame()
    df = df[df['code'] == code]
    for (subject, code), group in df.groupby(['subject_id', 'code']):
        if group.empty:
            continue
            
        # Process and filter measurements
        x, y, time = process_dataframe(group)
        x, y, time = filter_measurements(x, y, time, min_gap, min_tests, year_cutoff)
        
        
        if x is None:
            continue
        
        x, y, time = cluster_measurements(x, y, time)
        sub_df = pd.DataFrame()
        # sub_df['x'] = x.flatten()
        sub_df['numeric_value'] = y.flatten()
        sub_df['time'] = time.flatten()
        sub_df['subject_id'] = subject
        sub_df['code'] = code
        sub_df['cluster_size'] = len(y)
        filtered_df = pd.concat([filtered_df, sub_df])
        
    mu_global, std_global, min_global, max_global = calculate_global_reference_intervals(code)
    
    filtered_df['ref_mean'] = mu_global
    filtered_df['ref_std'] = std_global
    filtered_df['ref_lower'] = min_global
    filtered_df['ref_upper'] = max_global
    
    
    # for each individual, filter by time gap and minimum tests 
    return filtered_df

def reference_intervals_sex(df, code):
    """Calculate reference intervals for the given code and sex."""
    min_f, max_f = CBC_REFERENCE_INTERVALS[code]['F'][0], CBC_REFERENCE_INTERVALS[code]['F'][1]
    min_m, max_m = CBC_REFERENCE_INTERVALS[code]['M'][0], CBC_REFERENCE_INTERVALS[code]['M'][1]
    mu_f, mu_m = (min_f + max_f) / 2, (min_m + max_m) / 2
    
    df['ref_mean'] = df['gender'].apply(lambda x: mu_f if x == 'F' else mu_m)
    df['ref_lower'] = df['gender'].apply(lambda x: min_f if x == 'F' else min_m)
    df['ref_upper'] = df['gender'].apply(lambda x: max_f if x == 'F' else max_m)
    return df

def calculate_global_reference_intervals(code):
    """Calculate global reference intervals for the given code."""
    mu_global_female = CBC_REFERENCE_INTERVALS[code]['F']
    mu_global_male = CBC_REFERENCE_INTERVALS[code]['M']
    min_global = min(mu_global_female[0], mu_global_male[0])
    max_global = max(mu_global_female[1], mu_global_male[1])
    mu_global = (min_global + max_global) / 2
    std_global = (max_global - min_global) / (2 * 1.96)
    return mu_global, std_global, min_global, max_global

def check_normality(df):
    """Plot distribution and perform normality tests on numeric values."""
    #sns.histplot(df['numeric_value'], kde=True)
    #plt.show()

    # Shapiro-Wilk Test
    _, p_shapiro = stats.shapiro(df['numeric_value'])
    print("Shapiro-Wilk p-value:", p_shapiro)

    # D'Agostino and Pearson's Test
    _, p_dagostino = stats.normaltest(df['numeric_value'])
    print("D'Agostino-Pearson p-value:", p_dagostino)

    # Calculate and return mean and std
    mean = df['numeric_value'].mean()
    std = df['numeric_value'].std()
    return mean, std

def build_hierarchical_bayesian_model(df, samples=1000, sigma_pos=True):
    """Build and fit a hierarchical Bayesian model."""
    df['subject_code'] = pd.Categorical(df['subject_id']).codes
    values = df['numeric_value'].values
    subject_idx = df['subject_code'].values
    n_subjects = df['subject_code'].nunique()

    inter_subject_std = values.std()
    
    mu_global = df['ref_mean'].values[0]
    std_global = df['ref_std'].values[0]
    
    if sigma_pos == True:
        with pm.Model() as model:
            mu_pop = pm.Normal("mu_pop", mu=mu_global, sigma=std_global)
            sigma_pop = pm.HalfNormal("sigma_pop", sigma=std_global)

            mu_subject = pm.Normal("mu_subject", mu=mu_pop, sigma=sigma_pop, shape=n_subjects)
            sigma_subject = pm.HalfNormal("sigma_subject", sigma=sigma_pop, shape=n_subjects)

            y_obs = pm.Normal("y_obs", mu=mu_subject[subject_idx], sigma=sigma_subject[subject_idx], observed=values)
            map_estimate = pm.find_MAP(vars=[mu_subject, sigma_subject])
            
            approx = pm.fit(method='advi', n=20000)
            trace_vi = approx.sample(samples)
            #trace = pm.sample(2000, return_inferencedata=True)

        results = compile_results(df, trace_vi, map_estimate, sigma_pos=sigma_pos)
    
    else:
        with pm.Model() as model:
            mu_pop = pm.Normal("mu_pop", mu=mu_global, sigma=std_global)
            sigma_pop = pm.HalfNormal("sigma_pop", sigma=std_global)

            mu_subject = pm.Normal("mu_subject", mu=mu_pop, sigma=sigma_pop, shape=n_subjects)

            y_obs = pm.Normal("y_obs", mu=mu_subject[subject_idx], observed=values)
            map_estimate = pm.find_MAP(vars=[mu_subject])

            approx = pm.fit(method='advi', n=20000)
            trace_vi = approx.sample(samples)

        results = compile_results(df, trace_vi, map_estimate, sigma_pos=sigma_pos)

    return results, trace_vi, map_estimate, model


def compile_results(df, trace_vi, map_estimate, sigma_pos=True):
    """Compile results from the Bayesian model into a DataFrame."""
    mu_samples = trace_vi.posterior['mu_subject'].stack(draws=("chain", "draw")).values
    posterior_mean = np.mean(mu_samples, axis=1)
    posterior_std = np.std(mu_samples, axis=1) if not sigma_pos else np.mean(trace_vi.posterior['sigma_subject'].stack(draws=("chain", "draw")).values, axis=1)
    percentiles = {p: np.percentile(mu_samples, p, axis=1) for p in [2.5, 10, 25, 50, 75, 90, 97.5]}
    
    map_mean = map_estimate['mu_subject']
    map_std = map_estimate.get('sigma_subject', 0)
    map_lower, map_upper = calculate_range(df, map_mean, map_std)

    results_dict = {
        "subject_id": df['subject_id'].unique(),
        "code": df['code'][0],
        "bayes_num_tests": df.groupby('subject_id')['numeric_value'].count().values,
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "posterior_lower": percentiles[2.5],
        "posterior_upper": percentiles[97.5],
        "posteriorIQR_lower": percentiles[25],
        "posteriorIQR_upper": percentiles[75],
        "posteriorIQR_mean": percentiles[50],
        "posterior80_lower": percentiles[10],
        "posterior80_upper": percentiles[90],
        "posterior80_mean": percentiles[50],
        "posterior95_lower": percentiles[2.5],
        "posterior95_upper": percentiles[97.5],
        "posterior95_mean": percentiles[50]
    }

    return pd.DataFrame(results_dict)

def plot_ranges(full_results):
    """Plot range comparisons for each subject."""
    df = full_results
    subset = df.sample(20, random_state=42)
    
    # Check for the presence of expected columns
    expected_cols = ['posterior95_mean', 'posterior95_lower', 'posterior95_upper', 
                     'posterior80_mean', 'posterior80_lower', 'posterior80_upper',
                     'posteriorIQR_mean', 'posteriorIQR_lower', 'posteriorIQR_upper']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in DataFrame: {missing_cols}")
        return

    range_types = ['ref', 'posterior', 'setpoint', 'posterior95', 'posterior80', 'posteriorIQR']
    cols = [f"{r}_{suffix}" for r in range_types for suffix in ['lower', 'upper', 'mean']]
    subset_long = subset.melt(id_vars=['subject_id'], value_vars=cols, var_name='range_type', value_name='range_value')

    subset_long['model'] = subset_long['range_type'].str.extract(r'^(.*?)_')[0]
    subset_long['bound'] = subset_long['range_type'].str.extract(r'_(.*)$')[0]
    subset_long.drop(columns=['range_type'], inplace=True)
    df_long = subset_long.pivot(index=['subject_id', 'model'], columns='bound', values='range_value').reset_index()

    model_name_map = {
        'ref': 'Population Reference Range',
        'setpoint': 'Individual Reference Range',
        'posterior': 'Hierarchical Bayes (Posterior CV)',
        'posterior95': 'Hierarchical Bayes (Posterior 95% CI)',
        'posterior80': 'Hierarchical Bayes (Posterior 80% CI)',
        'posteriorIQR': 'Hierarchical Bayes (Posterior IQR)'
    }

    df_long['model'] = df_long['model'].replace(model_name_map)

    desired_order = [
        'Population Reference Range',
        'Individual Reference Range',
        'Hierarchical Bayes (Posterior CV)',
        'Hierarchical Bayes (Posterior IQR)',
        'Hierarchical Bayes (Posterior 80% CI)',
        'Hierarchical Bayes (Posterior 95% CI)'
    ]

    df_long['model'] = pd.Categorical(df_long['model'], categories=desired_order, ordered=True)
    df_long = df_long.sort_values(['subject_id', 'model'])

    models = desired_order
    n_models = len(models)
    colors = sns.color_palette("husl", n_models)

    subject_ids = df_long['subject_id'].unique()
    n_subjects = len(subject_ids)

    ncols = 10
    nrows = int(np.ceil(n_subjects / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), sharey=True)
    num_tests_dict = dict(zip(full_results['subject_id'], full_results['bayes_num_tests']))

    axes = axes.flatten()

    for idx, subject_id in enumerate(subject_ids):
        ax = axes[idx]
        subject_data = df_long[df_long['subject_id'] == subject_id]

        for i, (model, color) in enumerate(zip(models, colors)):
            model_data = subject_data[subject_data['model'] == model]

            if not model_data.empty:
                yerr = [
                    model_data['mean'].values[0] - model_data['lower'].values[0],
                    model_data['upper'].values[0] - model_data['mean'].values[0]
                ]

                ax.errorbar(
                    i,
                    model_data['mean'],
                    yerr=np.array(yerr).reshape(2, 1),
                    fmt='o',
                    label=model if idx == 0 else "",
                    color=color,
                    capsize=5,
                    markersize=8
                )

        ax.set_title(f"Subject {subject_id}, \n Num Tests: {num_tests_dict[subject_id]}")
        ax.set_xticks([])
        ax.grid(True, alpha=0.3)

    for ax in axes[n_subjects:]:
        ax.set_visible(False)

    for ax in axes[-ncols:]:
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(models, rotation=45)
        ax.set_xticks([])

    fig.text(-0.005, 0.5, 'Range Value', va='center', rotation='vertical', fontsize=14)

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=model,
                          markerfacecolor=color, markersize=8) for model, color in zip(models, colors)]
    fig.legend(handles=handles, title="Model Type", loc='upper center', bbox_to_anchor=(0.5, 1.2), ncols=3, fontsize=12)
    plt.xticks([])
    plt.tight_layout()
    plt.show()
    return plt