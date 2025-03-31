import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_fake_data(num_subjects=1000, min_samples=5, max_samples=20):
    """
    Generate fake RBC data in DataFrame format
    
    Args:
        num_subjects: number of unique subjects
        min_samples: minimum samples per subject
        max_samples: maximum samples per subject
    
    Returns:
        DataFrame with columns: subject_id, numeric_value, code, time
    """
    data_rows = []
    start_date = datetime(2018, 1, 1)  # Start from 2018
    
    for subject_id in range(num_subjects):
        # Generate number of samples for this subject
        n_samples = random.randint(min_samples, max_samples)
        
        # Generate baseline RBC value (different for males/females)
        is_female = random.choice([0, 1])
        baseline = np.random.normal(4.5 if is_female else 5.0, 0.3)
        
        # Generate sequence
        current_value = baseline
        current_date = start_date + timedelta(days=random.randint(0, 365))
        
        for _ in range(n_samples):
            # Generate RBC value with realistic variation
            variation = np.random.normal(0, 0.2)
            current_value = 0.8 * current_value + 0.2 * baseline + variation
            current_value = np.clip(current_value, 3.5, 6.5)
            
            # Add row to data
            data_rows.append({
                'subject_id': f'SUBJECT_{subject_id:06d}',
                'numeric_value': round(current_value, 2),
                'code': 'RBC',
                'time': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                'demographics': is_female
            })
            
            # Move to next date (30-90 days later)
            days_to_add = random.randint(30, 90)
            current_date += timedelta(days=days_to_add)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # Sort by subject_id and time
    df = df.sort_values(['subject_id', 'time'])
    
    return df

if __name__ == "__main__":
    # Generate fake data
    print("Generating fake RBC data...")
    df = generate_fake_data(num_subjects=1000, min_samples=5, max_samples=20)
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Number of subjects: {df['subject_id'].nunique()}")
    print(f"Total samples: {len(df)}")
    print(f"Samples per subject (min): {df.groupby('subject_id').size().min()}")
    print(f"Samples per subject (max): {df.groupby('subject_id').size().max()}")
    print(f"Samples per subject (mean): {df.groupby('subject_id').size().mean():.1f}")
    print("\nRBC value statistics:")
    print(df['numeric_value'].describe())
    
    # Save to CSV
    output_file = "filtered_lab_code_data.csv"
    df.to_csv(output_file, sep='\t', index=False)
    print(f"\nData saved to {output_file}")
    
    # Show example data
    print("\nExample data:")
    print(df.head(10))