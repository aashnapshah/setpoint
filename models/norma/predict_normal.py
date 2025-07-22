# ============================================================================
# FORECAST UNCERTAINTY MODEL
# ============================================================================

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import random
import sys

sys.path.append('../../')
from process.config import REFERENCE_INTERVALS

# ============================================================================
# DEVICE SETUP
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda": print(torch.cuda.get_device_name(0))

# ============================================================================
# DATASET CREATION
# ============================================================================

def is_normal(row):
    sex = 'F' if row['gender_concept_id'] == 1 else 'M'
    test_name = row['test_name']
    ref_range = REFERENCE_INTERVALS[test_name][sex]
    return row['numeric_value'] > ref_range[0] and row['numeric_value'] < ref_range[1]

def extract_time_features(dt):
    return dt.year, dt.month, dt.day, dt.hour
    
def create_sequences(df):
    df['normal'] = df.apply(is_normal, axis=1)
    sequences = []
    max_len = 0

    for patient_id in df['subject_id'].unique():
        patient_data = df[df['subject_id'] == patient_id].copy()
        patient_data = patient_data.sort_values('time')

        if patient_data['time'].dtype == 'object':
            patient_data['time'] = pd.to_datetime(patient_data['time'])

        times = patient_data['time'].tolist()
        values = patient_data['numeric_value'].values
        normal = patient_data['normal'].values
        
        if len(values) >= 3:  # Reduced minimum length requirement
            context_times = times[:-1]
            context_values = values[:-1]
            context_normal = normal[:-1]
            target_value = values[-1]
            target_normal = normal[-1]

            y, m, d, h = zip(*[extract_time_features(t) for t in context_times])
            max_len = max(max_len, len(context_values))

            sequences.append({
                'patient_id': patient_id,
                'values': context_values,
                'context_times': context_times,
                'target': target_value,
                'year': y,
                'month': m, 
                'day': d,
                'hour': h,
                'forecast_time': times[-1],
                'normal': context_normal,
                'target_normal': target_normal
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
        
        # Values and health status
        values = torch.tensor(sample['values'], dtype=torch.float32).unsqueeze(-1)
        normal_status = torch.tensor(sample['normal'], dtype=torch.float32).unsqueeze(-1)
        
        # Combine values and health status into single input
        x = torch.cat([values, normal_status], dim=-1)  # [seq_len, 2]
        
        pad_len = MAX_SEQ_LEN - len(x)
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(pad_len, 2)], dim=0)

        time_dict = {
            "year": torch.tensor(sample['year'], dtype=torch.long),
            "month": torch.tensor(sample['month'], dtype=torch.long),
            "day": torch.tensor(sample['day'], dtype=torch.long),
            "time": torch.tensor(sample['hour'], dtype=torch.long)
        }
        for k in time_dict:
            if pad_len > 0:
                time_dict[k] = torch.cat([time_dict[k], torch.zeros(pad_len, dtype=torch.long)], dim=0)

        y, m, d, h = extract_time_features(sample['forecast_time'])
        forecast_time_dict = {
            "year": torch.tensor([y], dtype=torch.long),
            "month": torch.tensor([m], dtype=torch.long),
            "day": torch.tensor([d], dtype=torch.long),
            "time": torch.tensor([h], dtype=torch.long)
        }
        next_target = torch.tensor(sample['target'], dtype=torch.float32)
        target_health = torch.tensor(sample['target_normal'], dtype=torch.float32)
        
        return x, time_dict, forecast_time_dict, next_target, target_health


def build_datasets(data_path, batch_size=16, test_size=0.2, val_size=0.1, random_state=42, sample_frac=1.0):
    """Build train, validation, and test datasets"""
    measurements = pd.read_csv(data_path, parse_dates=['time']).query('test_name == "HGB"').sample(frac=sample_frac)
    sequences = create_sequences(measurements)
    print(sequences[0])
    
    train_data, temp_data = train_test_split(sequences, test_size=test_size, random_state=random_state)
    val_data, test_data = train_test_split(temp_data, test_size=val_size/(test_size), random_state=random_state)
    
    train_dataset = HGBTimeSeriesDataset(train_data)
    val_dataset = HGBTimeSeriesDataset(val_data)
    test_dataset = HGBTimeSeriesDataset(test_data)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader, MAX_SEQ_LEN, test_data

# ============================================================================
# MODEL CREATION
# ============================================================================

# --- Positional Encoding Modules ---
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

# --- Transformer Encoder  ---
class TransformerEncoder(nn.Transformer):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super().__init__(
            d_model=ninp, nhead=nhead, dim_feedforward=nhid,
            num_encoder_layers=nlayers, num_decoder_layers=1,
            batch_first=True
        )
        self.input_proj = nn.Linear(2, ninp)  # 2 features: value + health_status
        self.pos_encoder = DateTimeEmbedding(ntoken, ninp)
        self.layer_norm = nn.LayerNorm(ninp)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(ninp, 1) 
        self.combined_decoder = nn.Linear(ninp + 1, 1) 
        self.src_mask = None
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def init_weights(self):
        nn.init.zeros_(self.decoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.combined_decoder.bias)
        nn.init.xavier_uniform_(self.combined_decoder.weight)

    def forward(self, src, src_key_padding_mask, dates=None, forecast_time_dict=None, target_health=None, device=None):
        src = self.input_proj(src)  # [batch, seq_len, 2] -> [batch, seq_len, ninp]
        if self.src_mask is None or self.src_mask.size(0) != src.shape[1]:
            self.src_mask = self._generate_square_subsequent_mask(src.shape[1]).to(device)
        time_emb = self.pos_encoder(dates)
        src = self.dropout(src + time_emb)
        src = self.layer_norm(src)
        output = self.encoder(src, src_key_padding_mask=src_key_padding_mask, mask=self.src_mask)
        forecast_emb = self.pos_encoder(forecast_time_dict).squeeze(1)
        
        # Combine forecast embedding with target health status
        if target_health is not None:
            target_health = target_health.to(device).unsqueeze(-1)  
            combined_features = torch.cat([forecast_emb, target_health], dim=-1)  
            return self.combined_decoder(combined_features)  
        else:
            return self.decoder(forecast_emb)  

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch_idx, (x, time_dict, forecast_time_dict, target, target_health) in enumerate(train_loader):
            x, target, target_health = x.to(device), target.to(device), target_health.to(device)
            for k in time_dict:
                time_dict[k] = time_dict[k].to(device)
                forecast_time_dict[k] = forecast_time_dict[k].to(device)
            # Create mask based on the first dimension (values) - if value is 0, it's padding
            mask = (x[:, :, 0] == 0).to(device)
            
            optimizer.zero_grad()
            output = model(x, mask, time_dict, forecast_time_dict, target_health, device)
            loss = nn.MSELoss()(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, time_dict, forecast_time_dict, target, target_health in val_loader:
                x, target, target_health = x.to(device), target.to(device), target_health.to(device)
                for k in time_dict:
                    time_dict[k] = time_dict[k].to(device)
                    forecast_time_dict[k] = forecast_time_dict[k].to(device)
                mask = (x[:, :, 0] == 0).to(device)
                output = model(x, mask, time_dict, forecast_time_dict, target_health, device)
                loss = nn.MSELoss()(output.squeeze(), target)
                val_losses.append(loss.item())

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}")

def evaluate(model, loader, sequences, device='cpu', save_path=None):
    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, (x, time_dict, forecast_time_dict, target, target_health) in enumerate(loader):
            x, target, target_health = x.to(device), target.to(device), target_health.to(device)
            for k in time_dict:
                time_dict[k] = time_dict[k].to(device)
                forecast_time_dict[k] = forecast_time_dict[k].to(device)
            mask = (x[:, :, 0] == 0).to(device)
            
            # Test prediction for healthy condition (1.0)
            healthy_health = torch.ones_like(target_health)
            output_healthy = model(x, mask, time_dict, forecast_time_dict, healthy_health, device)
            predictions_healthy = output_healthy.squeeze().cpu().numpy()
            
            # Test prediction for unhealthy condition (0.0)
            unhealthy_health = torch.zeros_like(target_health)
            output_unhealthy = model(x, mask, time_dict, forecast_time_dict, unhealthy_health, device)
            predictions_unhealthy = output_unhealthy.squeeze().cpu().numpy()
            
            for i in range(len(predictions_healthy)):
                seq_idx = batch_idx * loader.batch_size + i
                if seq_idx < len(sequences):
                    sample = sequences[seq_idx]
                    patient_id = sample['patient_id']
                    values = sample['values']
                    context_times = sample['context_times']
                    forecast_time = sample['forecast_time']
                else:
                    patient_id = f"unknown_{seq_idx}"
                    values = []
                    context_times = []
                    forecast_time = None
                
                results.append({
                    'patient_id': patient_id,
                    'target': target[i].item(),
                    'prediction_healthy': predictions_healthy[i],
                    'prediction_unhealthy': predictions_unhealthy[i],
                    'actual_target_health': target_health[i].item(),
                    'input_values': str(values), 
                    'input_times': str(context_times),  
                    'forecast_time': forecast_time,
                    'num_input_points': len(values)
                })
    
    df = pd.DataFrame(results)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def build_datasets(data_path, batch_size=16, test_size=0.2, val_size=0.1, random_state=42, sample_frac=1.0):
    """Build train, validation, and test datasets"""
    measurements = pd.read_csv(data_path, parse_dates=['time']).query('test_name == "HGB"').sample(frac=sample_frac, random_state=random_state)
    sequences = create_sequences(measurements)
    print(sequences[0])
    
    train_data, temp_data = train_test_split(sequences, test_size=test_size, random_state=random_state)
    val_data, test_data = train_test_split(temp_data, test_size=val_size/(test_size), random_state=random_state)
    
    train_dataset = HGBTimeSeriesDataset(train_data)
    val_dataset = HGBTimeSeriesDataset(val_data)
    test_dataset = HGBTimeSeriesDataset(test_data)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader, MAX_SEQ_LEN, test_data

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    data_path = '../../data/processed/lab_measurements.csv'
    save_dir = '../../models/norma'
    results_dir = '../../results/norma'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    batch_size = 32
    epochs = 10
    learning_rate = 1e-3
    sample_frac = 0.1  
    
    print("=== Time Series Forecasting ===")
    print(f"Device: {device}")
    
    print("\nBuilding datasets...")
    train_loader, val_loader, test_loader, max_seq_len, test_data = build_datasets(
        data_path, batch_size=batch_size, test_size=0.2, val_size=0.1, 
        random_state=42, sample_frac=sample_frac
    )
    print(f"Max sequence length: {max_seq_len}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = TransformerEncoder(
        ntoken=max_seq_len,
        ninp=128,
        nhead=8,
        nhid=256,
        nlayers=4,
        dropout=0.1
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTraining model...")
    train(model, train_loader, val_loader, epochs=epochs, lr=learning_rate, device=device)
    
    model_path = os.path.join(save_dir, 'transformer_model_health.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    print("\nEvaluating model...")
    test_results = evaluate(
        model, test_loader, test_data, device=device, 
        save_path=os.path.join(results_dir, 'test_predictions_health.csv')
    )
    
    mse_healthy = ((test_results['prediction_healthy'] - test_results['target']) ** 2).mean()
    mae_healthy = abs(test_results['prediction_healthy'] - test_results['target']).mean()
    
    mse_unhealthy = ((test_results['prediction_unhealthy'] - test_results['target']) ** 2).mean()
    mae_unhealthy = abs(test_results['prediction_unhealthy'] - test_results['target']).mean()
    
    print(f"\nTest Results:")
    print(f"Healthy Predictions - MSE: {mse_healthy:.4f}, MAE: {mae_healthy:.4f}")
    print(f"Unhealthy Predictions - MSE: {mse_unhealthy:.4f}, MAE: {mae_unhealthy:.4f}")
    
    summary_stats = {
        'mse_healthy': mse_healthy,
        'mae_healthy': mae_healthy,
        'mse_unhealthy': mse_unhealthy,
        'mae_unhealthy': mae_unhealthy,
        'num_test_samples': len(test_results),
        'model_parameters': sum(p.numel() for p in model.parameters())
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(os.path.join(results_dir, 'model_summary_health.csv'), index=False)
    print(f"\nResults saved to: {results_dir}")
    print("Files created:")
    print("- test_predictions_health.csv: Detailed predictions")
    print("- model_summary_health.csv: Model performance summary")

if __name__ == "__main__":
    main()
