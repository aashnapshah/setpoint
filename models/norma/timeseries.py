import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

class SinusoidalPositionalEmbedding(nn.Embedding):
    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out):
        n_pos, dim = out.shape
        position_enc = torch.tensor([
            [pos / (10000 ** (2 * (j // 2) / dim)) for j in range(dim)] for pos in range(n_pos)
        ])
        out.detach_()
        out.requires_grad = False
        out[:, :dim // 2] = torch.sin(position_enc[:, 0::2])
        out[:, dim // 2:] = torch.cos(position_enc[:, 1::2])
        return out

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(seq_len, device=self.weight.device)
        return super().forward(positions)


class DateTimeEmbedding(nn.Module):
    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__()
        self.embed_year = SinusoidalPositionalEmbedding(num_positions, embedding_dim, padding_idx)
        self.embed_month = nn.Embedding(13, embedding_dim)
        self.embed_day = nn.Embedding(32, embedding_dim)
        self.embed_time = nn.Embedding(25, embedding_dim)

    def forward(self, input):
        year = self.embed_year(input["year"])
        month = self.embed_month(input["month"])
        day = self.embed_day(input["day"])
        hour = self.embed_time(input["time"])
        return year + month + day + hour


def extract_time_features(dt):
    return dt.year, dt.month, dt.day, dt.hour

def create_sequences(df):
    """Create patient sequences from measurements"""
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

            seq_len = len(values) - 1
            max_len = max(max_len, seq_len)

            sequences.append({
                'patient_id': patient_id,
                'values': values[:seq_len],
                'target': values[-1],
                'year': years[:seq_len],
                'month': months[:seq_len],
                'day': days[:seq_len],
                'hour': hours[:seq_len]
            })

    global MAX_SEQ_LEN
    MAX_SEQ_LEN = max_len
    print(f"📏 Max sequence length: {MAX_SEQ_LEN}")

    return sequences


class HGBTimeSeriesDataset(Dataset):
    def __init__(self, sequences, embedding_dim=64):
        self.samples = sequences
        self.time_encoder = DateTimeEmbedding(3000, embedding_dim)  # 3000 years max
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        x = torch.tensor(sample['values'], dtype=torch.float32).unsqueeze(-1)  # [seq_len, 1]
        
        # Pad to max length if needed
        if len(x) < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - len(x)
            x = torch.cat([x, torch.zeros(pad_len, 1)], dim=0)
        
        # Create time dictionary
        time_dict = {
            "year": torch.tensor(sample['year'], dtype=torch.long).unsqueeze(0),
            "month": torch.tensor(sample['month'], dtype=torch.long).unsqueeze(0),
            "day": torch.tensor(sample['day'], dtype=torch.long).unsqueeze(0),
            "time": torch.tensor(sample['hour'], dtype=torch.long).unsqueeze(0)
        }
        
        # Pad time info if needed
        if len(time_dict["year"].squeeze(0)) < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - len(time_dict["year"].squeeze(0))
            for key in time_dict:
                time_dict[key] = torch.cat([time_dict[key], torch.zeros(1, pad_len, dtype=torch.long)], dim=1)
        
        dates = self.time_encoder(time_dict).squeeze(0)  # [seq_len, embedding_dim]
        next_target = torch.tensor(sample['target'], dtype=torch.float32)
        
        return x, dates, next_target


class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, nhead, nhid, nlayers, num_positions, dropout=0.1):
        super().__init__()
        self.time_encoder = DateTimeEmbedding(num_positions, input_dim)
        self.input_proj = nn.Linear(1, input_dim)  # Project single value to embedding
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.output_layer = nn.Linear(input_dim, 1)  # Predict single value

    def forward(self, x, time_info):
        """
        x: [batch_size, seq_len, 1] - input measurements
        time_info: dict with keys 'year', 'month', 'day', 'time' of shape [batch_size, seq_len]
        """
        # Project input values to embedding space
        x = self.input_proj(x)  # [batch, seq_len, input_dim]
        
        # Add time embeddings
        time_emb = self.time_encoder(time_info)  # [batch, seq_len, input_dim]
        x = x + time_emb
        
        # Transformer processing
        x = self.transformer(x)  # [batch, seq_len, input_dim]
        
        # Predict last step
        return self.output_layer(x[:, -1, :])  # [batch, 1]

def build_dataset(data_path):
    """Build the time series dataset"""
    print("📊 Loading data...")
    measurements = pd.read_csv(
        data_path,
        parse_dates=['measurement_time'],
        date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')
    ).query('test_name == "HGB"')
    sequences = create_sequences(measurements)
    print(f"✅ Created {len(sequences)} sequences")
    
    dataset = HGBTimeSeriesDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    print(f"\n🔍 Testing dataloader...")
    for i, (x, dates, target) in enumerate(dataloader):
        print(f"Batch {i+1}")
        print(f"  x: {x.shape}, dates: {dates.shape}, target: {target.shape}")
        print(f"  Patient ID: {sequences[i]['patient_id']}")  # Print patient ID
        print(f"  Sample values: {x[0, :5, 0].numpy()}")  # First 5 values of first sample
        print(f"  Target value: {target[0].item()}")  # First target value
        if i == 0: break

    return dataset, dataloader


def main():
    # Build dataset
    dataset, dataloader = build_dataset('../../data/processed/lab_measurements.csv')
    

if __name__ == "__main__":
    main()