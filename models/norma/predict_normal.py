# ============================================================================
# FORECAST UNCERTAINTY MODEL
# ============================================================================

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score
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
    test_vocab = {test_name: i for i, test_name in enumerate(df['test_name'].unique())}

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
        gender = patient_data['gender_concept_id'].values[0]
        test_name = patient_data['test_name'].values[0]

        if len(values) >= 3:
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
                'target_normal': target_normal,
                'sex': gender,
                'test_name': test_name
            })

    global TEST_VOCAB
    global MAX_SEQ_LEN
    MAX_SEQ_LEN = max_len
    TEST_VOCAB = test_vocab
    return sequences    

class HGBTimeSeriesDataset(Dataset):
    def __init__(self, sequences, test_vocab):
        self.samples = sequences
        self.test_vocab = test_vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        values = torch.tensor(sample['values'], dtype=torch.float32).unsqueeze(-1)
        normal_status = torch.tensor(sample['normal'], dtype=torch.float32).unsqueeze(-1)
        x = torch.cat([values, normal_status], dim=-1)

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

        sex = torch.tensor(sample['sex'], dtype=torch.long)
        test_id = torch.tensor(self.test_vocab[sample['test_name']], dtype=torch.long)

        return x, time_dict, forecast_time_dict, next_target, target_health, sex, test_id
        
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

# import torch
# import torch.nn as nn

# class GaussianSampler(nn.Module):
#     def __init__(self):
#         super(GaussianSampler, self).__init__()

#     def forward(self, mu, logvar):
#         """
#         Parameters:
#             mu: Tensor of means
#             logvar: Tensor of log-variances

#         Returns:
#             A sample from N(mu, sigma^2) using the reparameterization trick
#         """
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

# --- Causal Transformer with Health Embedding + Auxiliary Loss ---
class CausalTransformer(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, health_emb_dim=16, num_tests=10, loss_type='nll'):
        super().__init__()
        self.input_proj = nn.Linear(2, ninp) # INPUT, OUTPUT VALUES
        self.pos_encoder = DateTimeEmbedding(ntoken, ninp)
        self.layer_norm = nn.LayerNorm(ninp)
        self.dropout = nn.Dropout(dropout)

        encoder_layers = nn.TransformerEncoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)

        # Static covariate embeddings
        self.health_embedding = nn.Embedding(2, health_emb_dim)
        self.sex_embedding = nn.Embedding(2, 8)  # 0=male, 1=female
        self.test_embedding = nn.Embedding(num_tests, 8)
        
        # Combined decoder with static covariates
        total_emb_dim = health_emb_dim + 8 + 8  # health + sex + test
        
        self.loss_type = loss_type
        
        # Set the appropriate loss function as model attribute
        if loss_type == 'nll':
            self.loss_fn = nll_loss
            # For NLL: predict mean and log variance
            self.decoder_mean = nn.Linear(ninp + total_emb_dim, 1)
            self.decoder_logvar = nn.Linear(ninp + total_emb_dim, 1)
        elif loss_type == 'huber':
            self.loss_fn = nn.HuberLoss()
            # For single value prediction
            self.decoder = nn.Linear(ninp + total_emb_dim, 1)
            #self.decoder_mean = self.decoder
            #self.decoder_logvar = self.decoder
            # stuff into another thing 
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
            # For single value prediction
            self.decoder = nn.Linear(ninp + total_emb_dim, 1)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Auxiliary health classifier
        self.health_classifier = nn.Linear(ninp, 1)

    def generate_causal_mask(self, sz, device):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(device)

    def forward(self, src, src_key_padding_mask, dates=None, forecast_time_dict=None, target_health=None, sex=None, test_id=None, device=None):
        src = self.input_proj(src)
        time_emb = self.pos_encoder(dates)
        # MAKE SURE YOU ARE CONCATENATING THE TIME EMBEDDING TO THE INPUT VALUES not adding them
        src = self.dropout(src + time_emb)
        src = self.layer_norm(src)
        seq_len = src.size(1)
        causal_mask = self.generate_causal_mask(seq_len, device)
        encoded = self.transformer_encoder(src, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        summary = encoded[:, -1, :]
        forecast_emb = self.pos_encoder(forecast_time_dict).squeeze(1)

        # Static covariate embeddings
        if sex is not None:
            sex_emb = self.sex_embedding(sex)
        else:
            sex_emb = self.sex_embedding(torch.zeros(src.size(0), dtype=torch.long, device=device))
            
        if test_id is not None:
            test_emb = self.test_embedding(test_id)
        else:
            test_emb = self.test_embedding(torch.zeros(src.size(0), dtype=torch.long, device=device))

        if target_health is not None:
            health_idx = target_health.long()
            health_emb = self.health_embedding(health_idx)
            # Combine all embeddings: forecast + health + sex + test
            combined = torch.cat([forecast_emb, health_emb, sex_emb, test_emb], dim=-1)
            
            if self.loss_type == 'nll':
                # Predict mean and log variance
                mean = self.decoder_mean(combined)
                logvar = self.decoder_logvar(combined)
                logvar = torch.clamp(logvar, min=-10, max=10)  # Ensure numerical stability
                variance = torch.exp(logvar)
                
                # Auxiliary output
                health_logit = self.health_classifier(summary)
                return (mean, variance), health_logit
            else:
                # Predict single value
                pred = self.decoder(combined)
                
                # Auxiliary output
                health_logit = self.health_classifier(summary)
                return pred, health_logit
        else:
            health_emb = self.health_embedding(torch.zeros(src.size(0), dtype=torch.long, device=device))
            # Combine all embeddings: forecast + health + sex + test
            combined = torch.cat([forecast_emb, health_emb, sex_emb, test_emb], dim=-1)
            
            if self.loss_type == 'nll':
                # Predict mean and log variance
                mean = self.decoder_mean(combined)
                logvar = self.decoder_logvar(combined)
                logvar = torch.clamp(logvar, min=-10, max=10)  # Ensure numerical stability
                variance = torch.exp(logvar)
                
                # Auxiliary health classifier
                health_logit = self.health_classifier(summary)
                return (mean, variance), health_logit
            else:
                # Predict single value
                pred = self.decoder(combined)
                
                # Auxiliary health classifier
                health_logit = self.health_classifier(summary)
                return pred, health_logit

# import torch
# import torch.distributions as dist

# def gaussian_nll_loss(mu, std, x):
#     normal = dist.Normal(mu, std)
#     return -normal.log_prob(x).mean()


# MAKE YOUR OWN LOSS FUNCTION ( Z SCORE ????????)
def nll_loss(mean, variance, target):
    """
    Negative log-likelihood loss for Gaussian uncertainty estimation
    """
    # Ensure variance is positive and numerically stable
    variance = torch.clamp(variance, min=1e-6)
    
    # Negative log-likelihood of Gaussian
    # loss is the z score of your target - mean / var 
    # 0.5 * torch.log(2 * torch.pi * var) + (x - mu) ** 2 / (2 * var)
    loss = 0.5 * torch.log(2 * np.pi * variance) + 0.5 * ((target - mean) ** 2) / variance
    return loss.mean()

# --- Training with Auxiliary Loss ---
def train(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu', aux_weight=0.1):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_pred_losses = []
        train_health_losses = []
        for x, time_dict, forecast_time_dict, target, target_health, sex, test_id in train_loader:
            x, target, target_health = x.to(device), target.to(device), target_health.to(device)
            sex, test_id = sex.to(device), test_id.to(device)
            for k in time_dict:
                time_dict[k] = time_dict[k].to(device)
                forecast_time_dict[k] = forecast_time_dict[k].to(device)
            mask = (x[:, :, 0] == 0).to(device)

            optimizer.zero_grad()
            model_output = model(x, mask, time_dict, forecast_time_dict, target_health, sex, test_id)
            
            if model.loss_type == 'nll':
                (mean, variance), health_logits = model_output
                # Use model's loss function
                pred_loss = model.loss_fn(mean.squeeze(-1), variance.squeeze(-1), target)
            else:
                pred, health_logits = model_output
                # Use model's loss function
                pred_loss = model.loss_fn(pred.squeeze(-1), target)
            
            health_loss = nn.BCEWithLogitsLoss()(health_logits.squeeze(-1), target_health)
            loss = pred_loss + aux_weight * health_loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_pred_losses.append(pred_loss.item())
            train_health_losses.append(health_loss.item())

        # Validation
        model.eval()
        val_losses = []
        val_pred_losses = []
        val_health_losses = []
        with torch.no_grad():
            for x, time_dict, forecast_time_dict, target, target_health, sex, test_id in val_loader:
                x, target, target_health = x.to(device), target.to(device), target_health.to(device)
                sex, test_id = sex.to(device), test_id.to(device)
                for k in time_dict:
                    time_dict[k] = time_dict[k].to(device)
                    forecast_time_dict[k] = forecast_time_dict[k].to(device)
                mask = (x[:, :, 0] == 0).to(device)
                model_output = model(x, mask, time_dict, forecast_time_dict, target_health, sex, test_id)
                
                if model.loss_type == 'nll':
                    (mean, variance), health_logits = model_output
                    pred_loss = model.loss_fn(mean.squeeze(-1), variance.squeeze(-1), target)
                else:
                    pred, health_logits = model_output
                    pred_loss = model.loss_fn(pred.squeeze(-1), target)
                
                health_loss = nn.BCEWithLogitsLoss()(health_logits.squeeze(-1), target_health)
                loss = pred_loss + aux_weight * health_loss
                val_losses.append(loss.item())
                val_pred_losses.append(pred_loss.item())
                val_health_losses.append(health_loss.item())

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train - Total: {np.mean(train_losses):.4f}, Pred: {np.mean(train_pred_losses):.4f}, Health: {np.mean(train_health_losses):.4f}")
        print(f"  Val   - Total: {np.mean(val_losses):.4f}, Pred: {np.mean(val_pred_losses):.4f}, Health: {np.mean(val_health_losses):.4f}")


def evaluate(model, loader, sequences, device='cpu', save_path=None):
    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, (x, time_dict, forecast_time_dict, target, target_health, sex, test_id) in enumerate(loader):
            x, target, target_health = x.to(device), target.to(device), target_health.to(device)
            sex, test_id = sex.to(device), test_id.to(device)
            for k in time_dict:
                time_dict[k] = time_dict[k].to(device)
                forecast_time_dict[k] = forecast_time_dict[k].to(device)
            mask = (x[:, :, 0] == 0).to(device)
            
            # Test prediction for healthy condition (1.0)
            healthy_health = torch.ones_like(target_health)
            healthy_output, healthy_health_logits = model(x, mask, time_dict, forecast_time_dict, healthy_health, sex, test_id)
            
            # Test prediction for unhealthy condition (0.0)
            unhealthy_health = torch.zeros_like(target_health)
            unhealthy_output, unhealthy_health_logits = model(x, mask, time_dict, forecast_time_dict, unhealthy_health, sex, test_id)
                    
            if model.loss_type == 'nll':
                # Extract mean and std from NLL predictions
                mean_healthy, var_healthy = healthy_output
                mean_unhealthy, var_unhealthy = unhealthy_output
                
                pred_healthy = mean_healthy.squeeze(-1).cpu().numpy()
                std_healthy = torch.sqrt(var_healthy).squeeze(-1).cpu().numpy()
                pred_unhealthy = mean_unhealthy.squeeze(-1).cpu().numpy()
                std_unhealthy = torch.sqrt(var_unhealthy).squeeze(-1).cpu().numpy()
            else:
                # Single value predictions
                pred_healthy = healthy_output.squeeze(-1).cpu().numpy()
                pred_unhealthy = unhealthy_output.squeeze(-1).cpu().numpy()
                std_healthy = np.zeros_like(pred_healthy)  # No uncertainty for single value
                std_unhealthy = np.zeros_like(pred_unhealthy)
            
            for i in range(len(pred_healthy)):
                seq_idx = batch_idx * loader.batch_size + i
                if seq_idx < len(sequences):
                    sample = sequences[seq_idx]
                    patient_id = sample['patient_id']
                    values = sample['values']
                    context_normal = sample['normal']
                    context_times = sample['context_times']
                    forecast_time = sample['forecast_time']
                else:
                    patient_id = f"unknown_{seq_idx}"
                    values = []
                    context_times = []
                    forecast_time = None
                
                # Get the true prediction that matches the actual target health status
                if target_health[i].item() == 1:
                    true_pred = pred_healthy[i]
                else:
                    true_pred = pred_unhealthy[i]
                
                result_dict = {
                    'patient_id': patient_id,
                    'target': target[i].item(),
                    'pred_healthy': pred_healthy[i],
                    'pred_unhealthy': pred_unhealthy[i],
                    'actual_target_health': target_health[i].item(),
                    'true_prediction': true_pred,
                    'health_pred_healthy': torch.sigmoid(healthy_health_logits[i]).item(),
                    'health_pred_unhealthy': torch.sigmoid(unhealthy_health_logits[i]).item(),
                    'input_values': str(values), 
                    'input_times': str(context_times),  
                    'input_normal': str(context_normal),
                    'forecast_time': forecast_time,
                    'num_input_points': len(values)
                }
                
                if model.loss_type == 'nll':
                    result_dict.update({
                        'std_healthy': std_healthy[i],
                        'std_unhealthy': std_unhealthy[i]
                    })
                
                results.append(result_dict)
    
    df = pd.DataFrame(results)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def build_datasets(data_path, batch_size=16, test_size=0.2, val_size=0.1, random_state=42, sample_frac=1.0):
    """Build train, validation, and test datasets"""
    measurements = pd.read_csv(data_path, parse_dates=['time']).query('test_name == "HGB"').sample(frac=sample_frac, random_state=random_state)
    # convert nan to the string 'NA'
    measurements['test_name'] = measurements['test_name'].fillna('NA')
    sequences = create_sequences(measurements)
    print(sequences[0])
    
    train_data, temp_data = train_test_split(sequences, test_size=test_size, random_state=random_state)
    val_data, test_data = train_test_split(temp_data, test_size=val_size/(test_size), random_state=random_state)
    
    train_dataset = HGBTimeSeriesDataset(train_data, TEST_VOCAB)
    val_dataset = HGBTimeSeriesDataset(val_data, TEST_VOCAB)
    test_dataset = HGBTimeSeriesDataset(test_data, TEST_VOCAB)
    
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
    
    batch_size = 64
    epochs = 10
    learning_rate = 1e-2
    sample_frac = 0.3
    
    # Choose loss type: 'nll' for mean/variance prediction, 'huber' for single value
    loss_type = 'nll'  # Change this to 'huber' for single value prediction
    
    print("=== Time Series Forecasting ===")
    print(f"Device: {device}")
    print(f"Loss type: {loss_type.upper()}")
    
    print("\nBuilding datasets...")
    train_loader, val_loader, test_loader, max_seq_len, test_data = build_datasets(
        data_path, batch_size=batch_size, test_size=0.2, val_size=0.1, 
        random_state=42, sample_frac=sample_frac
    )
    print(f"Max sequence length: {max_seq_len}")
    print(f"Train sequences: {len(train_loader.dataset)}")
    print(f"Val sequences: {len(val_loader.dataset)}")
    print(f"Test sequences: {len(test_loader.dataset)}")
    # print(f"Train batches: {len(train_loader)}")
    # print(f"Val batches: {len(val_loader)}")
    # print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = CausalTransformer(
        ntoken=max_seq_len,
        ninp=128,
        nhead=8,
        nhid=256,
        nlayers=4,
        dropout=0.1,
        num_tests=len(TEST_VOCAB),
        loss_type=loss_type
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTraining model...")
    train(model, train_loader, val_loader, epochs=epochs, lr=learning_rate, device=device, aux_weight=1.0)
    
    model_suffix = model.loss_type
    model_path = os.path.join(save_dir, f'transformer_model_health_{model_suffix}_sample_{sample_frac}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    print("\nEvaluating model...")
    test_results = evaluate(
        model, test_loader, test_data, device=device, 
        save_path=os.path.join(results_dir, f'test_predictions_health_{model_suffix}_sample_{sample_frac}.csv')
    )
    
    # Calculate MSE, MAE, and R² for predictions vs true targets
    mse_healthy = ((test_results['pred_healthy'] - test_results['target']) ** 2).mean()
    mae_healthy = abs(test_results['pred_healthy'] - test_results['target']).mean()
    
    mse_unhealthy = ((test_results['pred_unhealthy'] - test_results['target']) ** 2).mean()
    mae_unhealthy = abs(test_results['pred_unhealthy'] - test_results['target']).mean()
    
    # Calculate R² using built-in function
    r2_healthy = r2_score(test_results['target'], test_results['pred_healthy'])
    r2_unhealthy = r2_score(test_results['target'], test_results['pred_unhealthy'])
    
    # Calculate overall MSE, MAE, and R² using the true_prediction column
    mse_overall = ((test_results['true_prediction'] - test_results['target']) ** 2).mean()
    mae_overall = abs(test_results['true_prediction'] - test_results['target']).mean()
    r2_overall = r2_score(test_results['target'], test_results['true_prediction'])
    
    print(f"\nTest Results:")
    print(f"Healthy Predictions - MSE: {mse_healthy:.4f}, MAE: {mae_healthy:.4f}, R²: {r2_healthy:.4f}")
    print(f"Unhealthy Predictions - MSE: {mse_unhealthy:.4f}, MAE: {mae_unhealthy:.4f}, R²: {r2_unhealthy:.4f}")
    print(f"Overall - MSE: {mse_overall:.4f}, MAE: {mae_overall:.4f}, R²: {r2_overall:.4f}")
    
    # Calculate AUC for health predictions
    try:
        # Use the health predictions from healthy condition (should be higher for actual healthy cases)
        health_scores = test_results['health_pred_healthy'].values
        health_labels = test_results['actual_target_health'].values
        
        # Calculate AUC
        auc_healthy = roc_auc_score(health_labels, health_scores)
        
        # Also calculate AUC for unhealthy condition predictions
        health_scores_unhealthy = test_results['health_pred_unhealthy'].values
        auc_unhealthy = roc_auc_score(health_labels, health_scores_unhealthy)
        
        print(f"Health Prediction AUC (Healthy condition): {auc_healthy:.4f}")
        print(f"Health Prediction AUC (Unhealthy condition): {auc_unhealthy:.4f}")
        
    except Exception as e:
        print(f"Could not calculate AUC: {e}")
        auc_healthy = 0.5
        auc_unhealthy = 0.5
    
    summary_stats = {
        'loss_type': model.loss_type,
        'mse_healthy': mse_healthy,
        'mae_healthy': mae_healthy,
        'r2_healthy': r2_healthy,
        'mse_unhealthy': mse_unhealthy,
        'mae_unhealthy': mae_unhealthy,
        'r2_unhealthy': r2_unhealthy,
        'mse_overall': mse_overall,
        'mae_overall': mae_overall,
        'r2_overall': r2_overall,
        'auc_healthy': auc_healthy,
        'auc_unhealthy': auc_unhealthy,
        'num_test_samples': len(test_results),
        'model_parameters': sum(p.numel() for p in model.parameters())
    }
    
    if model.loss_type == 'nll':
        # Add uncertainty metrics for NLL
        avg_std_healthy = test_results['std_healthy'].mean()
        avg_std_unhealthy = test_results['std_unhealthy'].mean()
        summary_stats.update({
            'avg_std_healthy': avg_std_healthy,
            'avg_std_unhealthy': avg_std_unhealthy
        })
        print(f"Average Uncertainty - Healthy: {avg_std_healthy:.4f}, Unhealthy: {avg_std_unhealthy:.4f}")
    
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(os.path.join(results_dir, f'model_summary_health_{model_suffix}_sample_{sample_frac}.csv'), index=False)
    print(f"\nResults saved to: {results_dir}")
    print("Files created:")
    print(f"- test_predictions_health_{model_suffix}_sample_{sample_frac}.csv: Detailed predictions")
    print(f"- model_summary_health_{model_suffix}_sample_{sample_frac}.csv: Model performance summary")

if __name__ == "__main__":
    main() 