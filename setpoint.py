import numpy as np
from sklearn.mixture import GaussianMixture
from datetime import datetime

def calculate_setpoint(x, y):
    """
    Estimate the setpoint of a time series defined by dates x and values y.
    """
    # Define basic parameters
    min_marker_gap = 90  # days
    min_tests = 5  # minimum number of markers

    # Convert to column vectors and clean data
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    
    # Remove NaN values
    idx = ~np.isnan(y)
    x = x[idx]
    y = y[idx]
    
    # Remove duplicate dates/times
    x, unique_idx = np.unique(x, return_index=True)
    y = y[unique_idx]

    # Sort by date
    sorted_idx = np.argsort(x.flatten())
    x = x[sorted_idx]
    y = y[sorted_idx]

    # Convert datetime to numeric days if necessary
    if isinstance(x[0], datetime):
        x = (np.array(x) - x[0]).astype('timedelta64[D]').astype(int)
    
    # Limit to markers with the desired minimum spacing gap
    nearest_marker = np.full(len(x), np.nan)
    for i in range(len(x)):
        # Calculate distances excluding the current element
        distances = np.abs(x[i] - np.setdiff1d(x, x[i]))
        nearest_marker[i] = np.min(distances)
    x = x[nearest_marker > min_marker_gap]
    y = y[nearest_marker > min_marker_gap]
    
    # Check if there are enough valid data points
    if len(x) < min_tests:
        raise ValueError(f"Less than {min_tests} data points are valid")
    
    # Fit Gaussian Mixture Models and calculate AIC for 1, 2, and 3 components
    aic = np.full(3, np.inf)
    mdl = [None] * 3
    
    # Convert to numpy array first
    y_array = np.array(y)
    
    for k in range(1, 4):
        try:
            # Reshape the data inside the loop
            y_reshaped = y_array.reshape(-1, 1)
            
            gmm = GaussianMixture(
                n_components=k,
                max_iter=1000,
                reg_covar=1e-6,
                random_state=42,
                init_params='kmeans'
            )
            gmm.fit(y_reshaped)
            mdl[k-1] = gmm
            aic[k-1] = gmm.aic(y_reshaped)
        except Exception as e:
            print(f"Error fitting GMM with {k} components: {e}")
            print(f"Error fitting GMM with {k} components")
            continue

    # Fallback: use simple statistics (mean and std) by default
    setpoint = np.mean(y)
    setpoint_cv = np.std(y) / setpoint

    # Use GMM results if a dominant multi-component model exists
    if not all(m is None for m in mdl):
        best_idx = np.argmin(aic)  # zero-indexed best model
        best_model = mdl[best_idx]
        mdl_prop = best_model.weights_
        model_num = best_idx + 1  # Adjust to 1-indexing to match MATLAB logic
        if (model_num == 2 and np.max(mdl_prop) > 0.7) or (model_num == 3 and np.max(mdl_prop) > 0.45):
            max_idx = np.argmax(mdl_prop)
            setpoint = best_model.means_[max_idx][0]
            setpoint_cv = np.sqrt(best_model.covariances_[max_idx][0]) / setpoint

    return setpoint, setpoint_cv[0]
