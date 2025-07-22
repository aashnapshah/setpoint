import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def extract_time_features(dt):
    return dt.year, dt.month, dt.day, dt.hour

def create_sequences(df, target_days_ahead=30):
    """
    Create sequences with target time point prediction
    target_days_ahead: How many days in the future to predict
    """
    sequences = []
    max_len = 0
    
    for patient_id in df['subject_id'].unique():
        patient_data = df[df['subject_id'] == patient_id].copy()
        patient_data = patient_data.sort_values('time')
        
        if patient_data['time'].dtype == 'object':
            patient_data['time'] = pd.to_datetime(patient_data['time'])
        
        times = patient_data['time'].tolist()
        values = patient_data['numeric_value'].values

        if len(values) >= 2:
            years, months, days, hours = [], [], [], []
            for t in times:
                y, m, d, h = extract_time_features(t)
                years.append(y)
                months.append(m)
                days.append(d)
                hours.append(h)

            # For each sequence, predict value at a specific future time
            for i in range(len(values) - 1):
                seq_len = i + 1
                max_len = max(max_len, seq_len)
                
                # Calculate target time (e.g., 30 days after last measurement)
                last_time = times[i]
                target_time = last_time + pd.Timedelta(days=target_days_ahead)
                
                # Find actual value closest to target time (if exists)
                target_value = None
                for j in range(i + 1, len(values)):
                    if abs((times[j] - target_time).days) <= 7:  # Within 7 days
                        target_value = values[j]
                        break
                
                if target_value is not None:
                    sequences.append({
                        'patient_id': patient_id,
                        'values': values[:seq_len],
                        'target': target_value,
                        'year': years[:seq_len],
                        'month': months[:seq_len],
                        'day': days[:seq_len],
                        'hour': hours[:seq_len],
                        'target_time': target_time
                    })

    global MAX_SEQ_LEN
    MAX_SEQ_LEN = max_len
    return sequences

class HGBTimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.samples = sequences

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        x = torch.tensor(sample['values'], dtype=torch.float32).unsqueeze(-1)
        
        if len(x) < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - len(x)
            x = torch.cat([x, torch.zeros(pad_len, 1)], dim=0)
        
        time_dict = {
            "year": torch.tensor(sample['year'], dtype=torch.long),
            "month": torch.tensor(sample['month'], dtype=torch.long),
            "day": torch.tensor(sample['day'], dtype=torch.long),
            "time": torch.tensor(sample['hour'], dtype=torch.long)
        }
        
        if len(time_dict["year"]) < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - len(time_dict["year"])
            for key in time_dict:
                time_dict[key] = torch.cat([time_dict[key], torch.zeros(pad_len, dtype=torch.long)], dim=0)
        
        next_target = torch.tensor(sample['target'], dtype=torch.float32)
        return x, time_dict, next_target

def build_datasets(data_path, batch_size=16, test_size=0.2, val_size=0.1, random_state=42, sample_frac=1.0):
    """Build train, validation, and test datasets"""
    measurements = pd.read_csv(data_path, parse_dates=['time']).query('test_name == "HGB"').sample(frac=sample_frac)
    sequences = create_sequences(measurements)
    
    # Split data
    train_data, temp_data = train_test_split(sequences, test_size=test_size, random_state=random_state)
    val_data, test_data = train_test_split(temp_data, test_size=val_size/(test_size), random_state=random_state)
    
    # Create datasets
    train_dataset = HGBTimeSeriesDataset(train_data)
    val_dataset = HGBTimeSeriesDataset(val_data)
    test_dataset = HGBTimeSeriesDataset(test_data)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader, MAX_SEQ_LEN
