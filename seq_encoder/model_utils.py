import os
import argparse
import json
import random
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
import math
import copy
import torch
import torch.nn as nn
from scipy import stats
import sys
sys.path.insert(0, '../') # Add our own scripts

from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error



DATE_PADDING = "0001-01-01 00:00:00"


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./')
    parser.add_argument('--output_dir', type=str, default='./outputs/')
    parser.add_argument('--save_prefix', type=str, default='')
    parser.add_argument('--code', type=str, default='RBC')
    
    # Model arguments
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--model_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # Training arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--temporal_len', type=int, default=10)
    
    # Other arguments
    parser.add_argument('--save_preds', action='store_true')
    parser.add_argument('--time_skip', action='store_true')
    parser.add_argument('--split', type=str, default='all')
    
    args = parser.parse_args()
    return args

def set_seed(seed):
    print("Setting seed:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_hparams(args):
    hparams = {
        "code": args.code,
        "input_dim": args.input_dim,
        "model_dim": args.model_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "temporal_len": args.temporal_len,
        "split": args.split
    }
    return hparams
def calc_metrics(true, pred, loss_fn):
    """Calculate regression metrics"""
    true = true.numpy()
    pred = pred.numpy()
    #print(true)
    #print(pred)
    return {
        "r2": r2_score(true, pred),
        "mae": mean_absolute_error(true, pred),
        "mape": mean_absolute_percentage_error(true, pred)
    }

def get_split_data(df, train_ratio=0.8, val_ratio=0.1, random_state=42):
    """
    Split data by subject ID into train/val/test sets
    """
    # Get unique subject IDs
    subject_ids = df['subject_id'].unique()
    np.random.seed(random_state)
    np.random.shuffle(subject_ids)
    
    # Calculate split indices
    n_subjects = len(subject_ids)
    train_idx = int(n_subjects * train_ratio)
    val_idx = int(n_subjects * (train_ratio + val_ratio))
    
    # Split subject IDs
    train_subjects = subject_ids[:train_idx]
    val_subjects = subject_ids[train_idx:val_idx]
    test_subjects = subject_ids[val_idx:]
    
    # Split DataFrame
    train_df = df[df['subject_id'].isin(train_subjects)]
    val_df = df[df['subject_id'].isin(val_subjects)]
    test_df = df[df['subject_id'].isin(test_subjects)]
    
    return (train_df, len(train_subjects)), (val_df, len(val_subjects)), (test_df, len(test_subjects))

def load_split_data(data_dir, hparams, data_type=None):
    """Load and process data into DataLoaders"""
    # Load processed data
    data_prefix = "filtered_lab_code_data"
    
    data = pd.read_csv(data_dir + data_prefix + ".csv", sep = "\t") # filtered_lab_data
    data_code = data[data["code"] == hparams["code"]]
    data_code = data_code[["code", "numeric_value", "subject_id", "time"]]
    
    # Split data
    train_df, val_df, test_df = get_split_data(data_code)
    
    def create_dataset(df):
        # Sort by subject and time
        df = df.sort_values(['subject_id', 'time'])
        grouped = df.groupby('subject_id')
        
        all_sequences = []
        sequence_lengths = []
        subject_ids = []
        for subject_id, subject_data in grouped:
            values = subject_data['numeric_value'].values
            times = pd.to_datetime(subject_data['time'])
            
            # Calculate time intervals in days
            time_intervals = np.zeros_like(values)
            time_intervals[1:] = np.array([
                (times.iloc[i] - times.iloc[i-1]).total_seconds() / (24*3600)
                for i in range(1, len(times))
            ])
            
            # Get demographics (assuming last digit of subject_id determines gender for fake data)
            demographics = np.full_like(values, int(subject_data['subject_id'].iloc[0][-1]) % 2)
            
            # Combine features
            sequence = np.column_stack([values, time_intervals, demographics])
            
            if len(sequence) >= 2:  # Need at least 2 points for prediction
                all_sequences.append(sequence)
                sequence_lengths.append(len(sequence))
                subject_ids.append(subject_id)
        
        # Pad sequences
        max_len = max(sequence_lengths)
        padded_data = np.zeros((len(all_sequences), max_len, 3))
        
        for i, seq in enumerate(all_sequences):
            padded_data[i, :len(seq)] = seq
        
        return torch.utils.data.TensorDataset(
            torch.tensor(padded_data, dtype=torch.float32),
            torch.tensor(sequence_lengths, dtype=torch.long),
            torch.tensor(subject_ids, dtype=torch.long)
        )
    
    # Create datasets
    train_dataset = create_dataset(train_df[0])
    val_dataset = create_dataset(val_df[0])
    test_dataset = create_dataset(test_df[0])
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        collate_fn=None  # Use default collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=hparams['batch_size'],
        shuffle=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=hparams['batch_size'],
        shuffle=False
    )
    
    return (train_loader, train_df[1]), (val_loader, val_df[1]), (test_loader, test_df[1])

  
def generate_realistic_rbc_data(num_samples=100, max_sequence_length=10):
    np.random.seed(42)
    
    sequence_lengths = np.random.randint(5, max_sequence_length + 1, size=num_samples)
    
    all_rbc_values = []
    all_time_intervals = []
    all_demographics = []
    
    for seq_len in sequence_lengths:
        # Generate baseline RBC value (different for males/females)
        is_female = np.random.choice([0, 1])
        baseline = np.random.normal(4.5 if is_female else 5.0, 0.3)
        
        # Generate sequence with realistic variations
        rbc_values = []
        current_value = baseline
        for _ in range(seq_len):
            variation = np.random.normal(0, 0.2)
            current_value = 0.8 * current_value + 0.2 * baseline + variation
            current_value = np.clip(current_value, 3.5, 6.5)
            rbc_values.append(current_value)
        
        time_intervals = np.random.randint(30, 90, size=seq_len)
        demographics = np.full(seq_len, is_female)
        
        all_rbc_values.append(rbc_values)
        all_time_intervals.append(time_intervals)
        all_demographics.append(demographics)
    
    # Create padded tensors
    max_len = max(sequence_lengths)
    padded_data = np.zeros((num_samples, max_len, 3))
    
    for i, (rbc, time, demo, seq_len) in enumerate(zip(all_rbc_values, 
                                                      all_time_intervals, 
                                                      all_demographics, 
                                                      sequence_lengths)):
        padded_data[i, :seq_len, 0] = rbc
        padded_data[i, :seq_len, 1] = time
        padded_data[i, :seq_len, 2] = demo
    
    return (torch.tensor(padded_data, dtype=torch.float32), 
            torch.tensor(sequence_lengths, dtype=torch.long))

def calculate_reference_range(model, input_sequence, confidence_level=0.95):
    with torch.no_grad():
        if not isinstance(input_sequence, torch.Tensor):
            input_sequence = torch.tensor(input_sequence, dtype=torch.float32)
        if len(input_sequence.shape) == 2:
            input_sequence = input_sequence.unsqueeze(0)
        
        mean, variance = model(input_sequence)
        mean = mean[:, -1]  # Get last prediction
        std = torch.sqrt(variance[:, -1])
        
        # Calculate range for given confidence level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        lower_bound = mean - z_score * std
        upper_bound = mean + z_score * std
        
        # Ensure ranges stay within physiologically possible values
        lower_bound = torch.clamp(lower_bound, min=2.5)
        upper_bound = torch.clamp(upper_bound, max=7.0)
        
        return lower_bound.item(), mean.item(), upper_bound.item()

def loop_batch(mode, batch_sz, split, loader, model, train_loader, temporal_len, loss_fn, optimizer=None, save_preds=True, time_skip=False, device=None, wandb=None):
    if mode == "train":
        model.train()
    else:
        model.eval()
    
    total_loss = 0
    individual_predictions = []  # Store predictions per person
    
    for batch in loader:
        batch_data, batch_lengths, batch_subject_ids = batch
        batch_data = batch_data.to(device)
        batch_lengths = batch_lengths.to(device)
        batch_size = batch_data.size(0)
        
        if mode == "train":
            optimizer.zero_grad()
        
        batch_loss = 0
        
        # Process each sequence in batch
        for i in range(batch_size):
            seq_len = batch_lengths[i]
            sequence = batch_data[i, :seq_len]
            subject_id = batch_subject_ids[i]  # Get subject ID
            
            # Use all but last value to predict last value
            input_seq = sequence[:-1].unsqueeze(0)
            target = sequence[-1, 0].unsqueeze(0)
            
            # Get predicted distribution
            mean, variance = model(input_seq)
            
            if mode == "train":
                loss = loss_fn(mean[:, -1], target, variance[:, -1])
                batch_loss += loss
            
            # Store individual's information with subject ID
            individual_predictions.append({
                'subject_id': subject_id,
                'history': sequence[:-1].cpu().numpy(),
                'true_next': target.cpu().numpy(),
                'predicted_mean': mean[:, -1].detach().cpu().numpy(),
                'predicted_variance': variance[:, -1].detach().cpu().numpy()
            })
        
        avg_batch_loss = batch_loss / batch_size
        
        if mode == "train":
            avg_batch_loss.backward()
            optimizer.step()
            
        total_loss += avg_batch_loss.item()
    
    # Calculate overall metrics
    all_targets = [p['true_next'] for p in individual_predictions]
    all_means = [p['predicted_mean'] for p in individual_predictions]
    metrics = calc_metrics(np.array(all_targets), np.array(all_means))
    
    if save_preds:
        all_predictions = {
            'individual_predictions': individual_predictions
        }
        with open(f'individual_predictions_{split}.json', 'w') as f:
            json.dump(all_predictions, f)
    else:
        all_predictions = None
    
    return total_loss, metrics, model, all_predictions