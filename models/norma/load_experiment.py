import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_experiment_results(filepath):
    """Load experiment results from pickle file"""
    with open(filepath, "rb") as f:
        experiment_data = pickle.load(f)
    
    return experiment_data

def analyze_experiment(experiment_data):
    """Analyze experiment results"""
    metadata = experiment_data['metadata']
    results = experiment_data['results']
    
    print("=== EXPERIMENT METADATA ===")
    print(f"Model Config: {metadata['model_config']}")
    print(f"Training Config: {metadata['training_config']}")
    print(f"Data Info: {metadata['data_info']}")
    
    print("\n=== RESULTS SUMMARY ===")
    for split in ['train_results', 'val_results', 'test_results']:
        if split in results:
            split_results = results[split]
            print(f"{split.replace('_', ' ').title()}:")
            print(f"  MSE: {split_results['mse']:.4f}")
            print(f"  MAE: {split_results['mae']:.4f}")
            print(f"  Number of predictions: {len(split_results['predictions'])}")
    
    return metadata, results

def plot_predictions_vs_targets(results, split_name):
    """Plot predictions vs targets for a given split"""
    predictions = results['predictions']
    targets = results['targets']
    
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.6)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'{split_name} Predictions vs Targets')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_uncertainty_analysis(results, split_name):
    """Plot uncertainty analysis if variance is available"""
    if 'variances' in results and any(v > 0 for v in results['variances']):
        predictions = results['predictions']
        targets = results['targets']
        variances = results['variances']
        
        # Calculate prediction errors
        errors = np.array(predictions) - np.array(targets)
        std_devs = np.sqrt(variances)
        
        plt.figure(figsize=(12, 4))
        
        # Plot 1: Error vs Standard Deviation
        plt.subplot(1, 2, 1)
        plt.scatter(std_devs, np.abs(errors), alpha=0.6)
        plt.xlabel('Predicted Standard Deviation')
        plt.ylabel('Absolute Error')
        plt.title('Uncertainty Calibration')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        plt.subplot(1, 2, 2)
        plt.hist(errors, bins=30, alpha=0.7, density=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Density')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'{split_name} Uncertainty Analysis')
        plt.tight_layout()
        plt.show()

def main():
    # Example usage
    experiment_dir = Path("experiments")
    
    if not experiment_dir.exists():
        print("No experiments directory found. Run training first.")
        return
    
    # Find the most recent experiment file
    experiment_files = list(experiment_dir.glob("experiment_output_*.pkl"))
    if not experiment_files:
        print("No experiment files found.")
        return
    
    latest_file = max(experiment_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading experiment: {latest_file}")
    
    # Load and analyze
    experiment_data = load_experiment_results(latest_file)
    metadata, results = analyze_experiment(experiment_data)
    
    # Plot results for each split
    for split_name in ['train_results', 'val_results', 'test_results']:
        if split_name in results:
            print(f"\n=== Plotting {split_name} ===")
            plot_predictions_vs_targets(results[split_name], split_name.replace('_', ' ').title())
            plot_uncertainty_analysis(results[split_name], split_name.replace('_', ' ').title())
    
    # Show sample metadata from predictions
    if 'test_results' in results:
        test_metadata = results['test_results']['metadata']
        print(f"\n=== Sample Test Metadata ===")
        for i, meta in enumerate(test_metadata[:5]):  # Show first 5 samples
            print(f"Sample {i+1}: {meta}")

if __name__ == "__main__":
    main() 