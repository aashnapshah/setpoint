import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')
from data_utils import TEST_VOCAB, CODE_TO_TEST_NAME

# Reference intervals (you may need to adjust these based on your data)
REFERENCE_INTERVALS = {
    'HGB': {'M': [14.0, 18.0], 'F': [12.0, 16.0]},
    'GLU': {'M': [70, 100], 'F': [70, 100]},
    'RBC': {'M': [4.5, 5.9], 'F': [4.1, 5.1]},
    'WBC': {'M': [4.0, 11.0], 'F': [4.0, 11.0]},
    'PLT': {'M': [150, 450], 'F': [150, 450]},
    # Add more as needed
}

def prepare_plot_data(predictions_df: pd.DataFrame, lab_measurements_path: str = "../../data/processed/lab_measurements.csv") -> pd.DataFrame:
    """
    Prepare data for plotting by merging predictions with lab measurements.
    
    Args:
        predictions_df: DataFrame with predictions
        lab_measurements_path: Path to lab measurements CSV
    
    Returns:
        Merged DataFrame ready for plotting
    """
    lab_measurements = pd.read_csv(lab_measurements_path)
    plot_data = predictions_df.copy()
    
    # Map test codes to names if needed
    if 'test_name' in plot_data.columns and plot_data['test_name'].dtype in ['int64', 'float64']:
        plot_data['test_name_actual'] = plot_data['test_name'].map(CODE_TO_TEST_NAME)
        plot_data['test_name'] = plot_data['test_name_actual']
    
    # Add lab measurement summary statistics
    if 'subject_id' in lab_measurements.columns:
        lab_measurements['time'] = pd.to_datetime(lab_measurements['time'])
        summary = lab_measurements.groupby(['subject_id', 'test_name']).agg(
            num_measurements=('time', 'size'),
            first_time=('time', 'first'),
            last_time=('time', 'last'), 
            observed_std=('numeric_value', 'std')
        ).reset_index()
        
        summary['time_length'] = (summary['last_time'] - summary['first_time']).dt.total_seconds() / (60*60*24)
        plot_data = summary.merge(plot_data, on=['subject_id', 'test_name'], how='right')
    
    return plot_data

def plot_prediction_analysis(test_data: pd.DataFrame, test_name: str, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive analysis plots for a single test.
    
    Args:
        test_data: DataFrame with test data
        test_name: Name of the test
        save_path: Optional path to save the plot
    
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    
    # Top plot - predictions vs actual
    ax0 = fig.add_subplot(gs[0, :])
    
    if 'condition' in test_data.columns:
        normal_mask = test_data['condition'] == 1
        abnormal_mask = test_data['condition'] == 0
        
        ax0.scatter(test_data[normal_mask]['prediction'],
                   test_data[normal_mask]['target'],
                   c='green', label='Normal', alpha=0.6, s=100)
        ax0.scatter(test_data[abnormal_mask]['prediction'],
                   test_data[abnormal_mask]['target'],
                   c='red', label='Abnormal', alpha=0.6, s=100)
    else:
        ax0.scatter(test_data['prediction'], test_data['target'], 
                   alpha=0.6, s=100, label='All samples')
    
    # Add diagonal line
    min_val = min(test_data['prediction'].min(), test_data['target'].min())
    max_val = max(test_data['prediction'].max(), test_data['target'].max())
    ax0.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect prediction')
    
    r2_overall = r2_score(test_data['target'], test_data['prediction'])
    
    ax0.set_xlabel('Predicted')
    ax0.set_ylabel('Observed')
    ax0.set_title(f'{test_name} - Prediction vs Target Value (n={len(test_data)} sequences)\nR²: {r2_overall:.3f}')
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    
    # Bottom left - Number of points vs uncertainty
    ax1 = fig.add_subplot(gs[1, 0])
    
    # Debug: check data availability
    has_num_measurements = 'num_measurements' in test_data.columns
    has_variance = 'variance' in test_data.columns
    variance_has_data = has_variance and (test_data['variance'] > 0).any()
    
    if has_num_measurements and has_variance and variance_has_data:
        if 'condition' in test_data.columns:
            normal_mask = test_data['condition'] == 1
            abnormal_mask = test_data['condition'] == 0
            
            # Filter out zero variance for plotting
            valid_data = test_data[test_data['variance'] > 0]
            if len(valid_data) > 0:
                normal_valid = valid_data[valid_data['condition'] == 1]
                abnormal_valid = valid_data[valid_data['condition'] == 0]
                
                if len(normal_valid) > 0:
                    ax1.scatter(normal_valid['num_measurements'], 
                               np.sqrt(normal_valid['variance']),
                               alpha=0.6, label='Normal', c='green', s=60)
                if len(abnormal_valid) > 0:
                    ax1.scatter(abnormal_valid['num_measurements'], 
                               np.sqrt(abnormal_valid['variance']),
                               alpha=0.6, label='Abnormal', c='red', s=60)
            else:
                ax1.text(0.5, 0.5, 'No valid variance data (all zeros)', 
                        ha='center', va='center', transform=ax1.transAxes)
        else:
            valid_data = test_data[test_data['variance'] > 0]
            if len(valid_data) > 0:
                ax1.scatter(valid_data['num_measurements'], np.sqrt(valid_data['variance']),
                           alpha=0.6, label='All samples', s=60)
            else:
                ax1.text(0.5, 0.5, 'No valid variance data (all zeros)', 
                        ha='center', va='center', transform=ax1.transAxes)
        
        ax1.set_xlabel('Number of Input Points')
        ax1.set_ylabel('Predicted Standard Deviation')
        ax1.legend()
        ax1.set_title(f'{test_name} - Uncertainty vs Number of Input Points')
        ax1.grid(True, alpha=0.3)
    else:
        # More detailed error message
        missing_info = []
        if not has_num_measurements:
            missing_info.append('num_measurements')
        if not has_variance:
            missing_info.append('variance')
        elif not variance_has_data:
            missing_info.append('non-zero variance')
            
        error_msg = f'Missing: {", ".join(missing_info)}'
        ax1.text(0.5, 0.5, f'Uncertainty data not available\n({error_msg})', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Uncertainty Analysis')
    
    # Bottom right - Observed std vs predicted std
    ax2 = fig.add_subplot(gs[1, 1])
    if 'observed_std' in test_data.columns and 'variance' in test_data.columns:
        if 'condition' in test_data.columns:
            normal_mask = test_data['condition'] == 1
            abnormal_mask = test_data['condition'] == 0
            
            ax2.scatter(test_data[normal_mask]['observed_std'], 
                       np.sqrt(test_data[normal_mask]['variance']),
                       alpha=0.5, label='Normal', c='green')
            ax2.scatter(test_data[abnormal_mask]['observed_std'], 
                       np.sqrt(test_data[abnormal_mask]['variance']),
                       alpha=0.5, label='Abnormal', c='red')
        else:
            ax2.scatter(test_data['observed_std'], np.sqrt(test_data['variance']),
                       alpha=0.5, label='All samples')
        
        ax2.set_xlabel('Observed Standard Deviation')
        ax2.set_ylabel('Predicted Standard Deviation')
        ax2.legend()
        ax2.set_title(f'{test_name} - Predicted vs Observed Uncertainty')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Observed std data not available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Std Comparison')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_overall_metrics(metrics_summary: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot overall performance metrics across all tests and splits.
    
    Args:
        metrics_summary: DataFrame with metrics summary
        save_path: Optional path to save the plot
    
    Returns:
        matplotlib Figure object
    """
    # Filter to overall metrics only
    overall_metrics = metrics_summary[metrics_summary['test_name'] == 'overall'].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    metrics_to_plot = ['r2', 'mae', 'mse', 'loss']
    colors = ['blue', 'orange', 'green']
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        if metric in overall_metrics.columns:
            splits = overall_metrics['split'].unique()
            values = [overall_metrics[overall_metrics['split'] == split][metric].iloc[0] 
                     for split in splits]
            
            bars = ax.bar(splits, values, color=colors[:len(splits)], alpha=0.7)
            ax.set_title(f'Overall {metric.upper()}')
            ax.set_ylabel(metric.upper())
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, f'{metric} not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Overall {metric.upper()}')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_per_test_metrics(metrics_summary: pd.DataFrame, metric: str = 'r2', save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot metrics for each test type across splits.
    
    Args:
        metrics_summary: DataFrame with metrics summary
        metric: Metric to plot ('r2', 'mae', 'mse')
        save_path: Optional path to save the plot
    
    Returns:
        matplotlib Figure object
    """
    # Filter out overall metrics
    per_test_metrics = metrics_summary[metrics_summary['test_name'] != 'overall'].copy()
    
    if len(per_test_metrics) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No per-test metrics available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Per-Test {metric.upper()}')
        return fig
    
    # Pivot data for plotting
    pivot_data = per_test_metrics.pivot(index='test_name', columns='split', values=metric)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create grouped bar plot
    pivot_data.plot(kind='bar', ax=ax, alpha=0.8)
    ax.set_title(f'Per-Test {metric.upper()} Across Splits')
    ax.set_ylabel(metric.upper())
    ax.set_xlabel('Test Name')
    ax.legend(title='Split')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_residuals_analysis(predictions_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot residuals analysis across all predictions.
    
    Args:
        predictions_df: DataFrame with predictions
        save_path: Optional path to save the plot
    
    Returns:
        matplotlib Figure object
    """
    # Calculate residuals
    residuals = predictions_df['target'] - predictions_df['prediction']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals vs Predictions
    axes[0, 0].scatter(predictions_df['prediction'], residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[0, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Residuals')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals by test type
    if 'test_name' in predictions_df.columns:
        test_names = predictions_df['test_name'].unique()
        if len(test_names) <= 10:  # Only plot if not too many tests
            predictions_df['residuals'] = residuals
            test_residuals = [predictions_df[predictions_df['test_name'] == test]['residuals'].values 
                            for test in test_names]
            
            axes[1, 1].boxplot(test_residuals, labels=test_names)
            axes[1, 1].set_xlabel('Test Name')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].set_title('Residuals by Test Type')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, f'Too many test types ({len(test_names)}) to display', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Residuals by Test Type')
    else:
        axes[1, 1].text(0.5, 0.5, 'Test type information not available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Residuals by Test Type')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def create_all_plots(
    metrics_summary: pd.DataFrame,
    test_predictions: pd.DataFrame,
    save_dir: str = "plots",
    wandb_module = None
) -> Dict[str, plt.Figure]:
    """
    Create all visualization plots and optionally log to wandb.
    
    Args:
        metrics_summary: DataFrame with comprehensive metrics
        test_predictions: DataFrame with test predictions
        save_dir: Directory to save plots
        wandb_module: wandb module for logging (optional)
    
    Returns:
        Dictionary of plot names to figure objects
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    figures = {}
    
    # 1. Overall metrics plot
    print("Creating overall metrics plot...")
    fig_overall = plot_overall_metrics(metrics_summary, f"{save_dir}/overall_metrics.png")
    figures['overall_metrics'] = fig_overall
    
    # 2. Per-test MAE plot
    print("Creating per-test MAE plot...")
    fig_mae = plot_per_test_metrics(metrics_summary, 'mae', f"{save_dir}/per_test_mae.png")
    figures['per_test_mae'] = fig_mae
    
    # 3. Individual test prediction analysis plots
    plot_data = prepare_plot_data(test_predictions)
    test_names = test_predictions['test_name'].unique() if 'test_name' in test_predictions.columns else []
    
    for test_name in test_names:
        test_data = plot_data[plot_data['test_name'] == test_name]
        if len(test_data) > 0:
            fig_test = plot_prediction_analysis(test_data, test_name, f"{save_dir}/prediction_analysis_{test_name}.png")
            figures[f'prediction_analysis_{test_name}'] = fig_test
    
    # Log to wandb if provided
    if wandb_module is not None:
        print("Logging plots to wandb...")
        for plot_name, fig in figures.items():
            wandb_module.log({f"plots/{plot_name}": wandb_module.Image(fig)})
    
    return figures
