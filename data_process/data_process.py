import os
import argparse
import pandas as pd
from typing import Dict, Union
import glob
import datasets

from processors.demographics import get_demographics_data, get_demographic_summary
from processors.measurements import get_bmi_data, get_bmi_statistics
from processors.cbc import get_cbc_data, get_cbc_subject_statistics

DEFAULT_PATH = '/Users/aashnashah/Desktop/ssh_mount/data/EHRSHOT/meds_omop_ehrshot/'
DEFAULT_MIN_TESTS = 5
DEFAULT_MIN_DAYS_BETWEEN = 0
DEFAULT_OUTPUT_DIR = "../data"

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process EHR data for ML tasks.")
    parser.add_argument("--raw", type=bool, default=False)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    return args

def load_dataset(data_path: str) -> pd.DataFrame:
    """Load dataset from Parquet files and convert to a Pandas DataFrame."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist")
    
    dataset = datasets.Dataset.from_parquet(os.path.join(data_path, 'data/*'))
    return dataset.to_pandas()

def save_csv(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
            output_dir: str, 
            name: str = None) -> None:
    """
    Save DataFrame(s) to CSV file(s).
    """
    if isinstance(data, pd.DataFrame) and name is None:
        raise ValueError("Name parameter is required when saving a single DataFrame")
    
    data_dict = {name: data} if isinstance(data, pd.DataFrame) else data
    
    os.makedirs(output_dir, exist_ok=True)
    
    for df_name, df in data_dict.items():
        filepath = os.path.join(output_dir, f"{df_name}.csv")
        df.to_csv(filepath, index=False)
        print(f"Saved {df_name} to {filepath}")

def get_dataset_statistics(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], name: str = "", stage: str = "") -> None:
    """
    Print statistics for dataset(s).
    """
    if stage:
        print(f"\n{stage} Dataset Statistics:")
    
    data_dict = {name or "Dataset": data} if isinstance(data, pd.DataFrame) else data
    
    for df_name, df in data_dict.items():
        print(f"\n{df_name} data:")
        print(f"# Events: {df.shape[0]}")
        print(f"# Unique subjects: {len(df['subject_id'].unique())}")
        print(f"# Average # Events per Patient: {df.groupby('subject_id').size().mean():.2f}")

def subset_patient_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Merge patient dataframes based on common subject IDs."""
    
    subject_sets = {name: set(df["subject_id"].unique()) for name, df in data_dict.items()}
    common_subjects = set.intersection(*subject_sets.values()) if subject_sets else set()
    
    return {
        name: df[df['subject_id'].isin(common_subjects)]
        for name, df in data_dict.items()
    }

def merge_patient_data(data_dict: Dict[str, pd.DataFrame], time_tolerance: pd.Timedelta = pd.Timedelta("4D")) -> pd.DataFrame:
    """Merge patient dataframes with flexibility in time matching and handling static demographics."""
    # Identify static (demographics) and time-series dataframes
    time_series_dfs = {name: df.copy() for name, df in data_dict.items() if "time" in df.columns}
    static_dfs = {name: df.copy() for name, df in data_dict.items() if "time" not in df.columns}

     # Convert time columns to datetime if needed
    for name in time_series_dfs:
        time_series_dfs[name]["time"] = pd.to_datetime(time_series_dfs[name]["time"])
        time_series_dfs[name] = time_series_dfs[name].sort_values(["subject_id", "time"])

    # Merge time-series data using nearest time matching
    merged_df = None
    for df in time_series_dfs.values():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge_asof(
                merged_df.sort_values(["subject_id", "time"]),
                df.sort_values(["subject_id", "time"]),
                on="time",
                by="subject_id",
                direction="nearest",
                tolerance=time_tolerance
            )

    # Merge static data (demographics) on subject_id
    for df in static_dfs.values():
        merged_df = merged_df.merge(df, on="subject_id", how="left")
    merged_df['Age'] = (merged_df['time'] - merged_df['DOB']).dt.days // 365
    return merged_df

def get_processed_data(name: str, process_fn, args, dependencies=None) -> pd.DataFrame:
    """Process data or load if exists."""
    processed_path = os.path.join(args.output_dir, "processed", f"{name}.csv")
    
    if not args.raw and os.path.exists(processed_path):
        print(f"Loading {name} data from {processed_path}")
        return pd.read_csv(processed_path)
    
    if not 'raw_df' in globals():
        globals()['raw_df'] = load_dataset(args.data_dir)
        get_dataset_statistics(raw_df, name="Raw", stage="Initial")
    
    print(f"\n Processing {name} data...")
    result = process_fn(raw_df, *dependencies) if dependencies else process_fn(raw_df)
    save_csv(result, args.output_dir + "/processed", name)
    return result

def main():
    args = parse_args()
            
    # Process data in stages to manage memory
    demo_df = get_processed_data("demographics", get_demographics_data, args=args)
    body_df = get_processed_data("body_measurements", get_bmi_data, args=args)
    cbc_df = get_processed_data("cbc_measurements", get_cbc_data, args=args, dependencies=[demo_df])
    
    # Merge datasets based on common subject IDs
    print("\nMerging Data:")
    data_dict = {
        "demographics": demo_df,
        "body_measurements": body_df,
        "cbc_measurements": cbc_df
    }

    filtered_data = subset_patient_data(data_dict)
    get_dataset_statistics(filtered_data, name="Processed", stage="Merged")
    save_csv(filtered_data, args.output_dir + "/processed")

    merged_data = merge_patient_data(filtered_data)
    save_csv(merged_data, args.output_dir + "/merged", name="all_processed_events.csv")
    print(merged_data.head())
    
    # Print summaries of merged data
    print("\nDemographics Summary:")
    print(get_demographic_summary(filtered_data["demographics"]))
    
    print("\nBMI Statistics:")
    print(get_bmi_statistics(filtered_data["body_measurements"]))
    
    print("\nCBC Statistics:")
    print(get_cbc_subject_statistics(filtered_data["cbc_measurements"]).head())

if __name__ == "__main__":
    main()