import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model import TimeConditionedTransformer
from data import create_dataloaders
from decoders import get_loss_fn
import pandas as pd
import argparse
import numpy as np
import os
import json
# -------------------------
# Loss functions
# -------------------------
# Now using get_loss_fn from decoders.py

# -------------------------
# Evaluation
# -------------------------
def evaluate(model, loader, loss_fn, device, decoder_type='nll'):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, t, c, sex, lab_code, query_t, query_c, y, pad_mask, subject_ids in loader:
            x, t, c = x.to(device), t.to(device), c.to(device)
            sex, lab_code = sex.to(device), lab_code.to(device)
            query_t, query_c = query_t.to(device), query_c.to(device)
            y = y.to(device)
            pad_mask = pad_mask.to(device)

            output = model(x, t, c, sex, lab_code, query_t, query_c, pad_mask)

            # Handle different decoder outputs
            if decoder_type == 'mdn':
                pi, mu, log_var = output
                loss = loss_fn(pi, mu, log_var, y)
            elif decoder_type == 'diffusion':
                # For diffusion, we need to add noise during evaluation
                noise = torch.randn_like(y)
                loss = loss_fn(output, noise)
            else:
                loss = loss_fn(output, y)

            total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

def save_predictions(model, loader, split_name, device, save_dir="predictions", decoder_type='nll'):
    """Save predictions for a dataset split as CSV"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    all_predictions = []
    all_variances = []
    all_targets = []
    all_patient_ids = []
    all_subject_ids = []
    all_test_names = []
    
    with torch.no_grad():
        for batch_idx, (x, t, c, sex, lab_code, query_t, query_c, y, pad_mask, subject_ids) in enumerate(loader):
            x, t, c = x.to(device), t.to(device), c.to(device)
            sex, lab_code = sex.to(device), lab_code.to(device)
            query_t, query_c = query_t.to(device), query_c.to(device)
            y = y.to(device)
            pad_mask = pad_mask.to(device)

            pred = model(x, t, c, sex, lab_code, query_t, query_c, pad_mask)
            
            # Handle different decoder outputs
            if decoder_type == 'mdn':
                pi, mu, log_var = pred
                # Use the component with highest weight as prediction
                best_component = torch.argmax(pi, dim=1)
                pred_mean = mu[torch.arange(len(mu)), best_component]
                pred_var = torch.exp(log_var[torch.arange(len(log_var)), best_component])
                all_predictions.extend(pred_mean.cpu().numpy().tolist())
                all_variances.extend(pred_var.cpu().numpy().tolist())
                pred_len = len(pred_mean)
            elif decoder_type == 'quantile':
                # Use median (0.5 quantile) as prediction
                median_idx = 4  # 0.5 quantile is at index 4 (0.1, 0.2, ..., 0.9)
                pred_mean = pred[:, median_idx]
                # Use interquartile range as uncertainty
                q75_idx, q25_idx = 6, 2  # 0.7 and 0.3 quantiles
                iqr = pred[:, q75_idx] - pred[:, q25_idx]
                all_predictions.extend(pred_mean.cpu().numpy().tolist())
                all_variances.extend((iqr/1.35)**2)  # Convert IQR to variance approximation
                pred_len = len(pred_mean)
            elif decoder_type == 'nll':
                pred_mean = pred[:, 0]  # Extract mean from (mean, log_var) output
                pred_var = torch.exp(pred[:, 1])  # Extract variance from log_var
                all_predictions.extend(pred_mean.cpu().numpy().tolist())
                all_variances.extend(pred_var.cpu().numpy().tolist())
                pred_len = len(pred_mean)
            else:  # standard, diffusion
                all_predictions.extend(pred.cpu().numpy().tolist())
                all_variances.extend([0.0] * len(pred))  # No variance for other losses
                pred_len = len(pred)
            all_targets.extend(y.cpu().numpy().tolist())
            all_patient_ids.extend([f"{split_name}_batch_{batch_idx}_sample_{i}" for i in range(pred_len)])
            all_subject_ids.extend(subject_ids)
            # Convert lab_code tensor to list of values
            lab_code_values = [lab_code[i].item() for i in range(len(lab_code))]
            all_test_names.extend(lab_code_values)
    
    # Create DataFrame and save as CSV
    results_df = pd.DataFrame({
        'patient_id': all_patient_ids,
        'subject_id': all_subject_ids,
        'test_name': all_test_names,
        'prediction': all_predictions,
        'variance': all_variances,
        'target': all_targets
    })
    
    results_df.to_csv(f"{save_dir}/{split_name}_predictions.csv", index=False)
    
    # Calculate metrics
    mse = np.mean((np.array(all_predictions) - np.array(all_targets)) ** 2)
    mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
    
    print(f"{split_name} - MSE: {mse:.4f}, MAE: {mae:.4f}")
    return {'mse': mse, 'mae': mae}

def generate_counterfactuals(model, loader, device, save_dir="predictions"):
    """Generate counterfactual predictions by switching conditions"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    counterfactual_data = []

    with torch.no_grad():
        for batch_idx, (x, t, c, sex, lab_code, query_t, query_c, y, pad_mask, subject_ids) in enumerate(loader):
            x, t, c = x.to(device), t.to(device), c.to(device)
            sex, lab_code = sex.to(device), lab_code.to(device)
            query_t, query_c = query_t.to(device), query_c.to(device)
            y = y.to(device)
            pad_mask = pad_mask.to(device)
            
            # Original prediction
            original_pred = model(x, t, c, sex, lab_code, query_t, query_c, pad_mask)
            if args.decoder_type == 'mdn':
                pi, mu, log_var = original_pred
                best_component = torch.argmax(pi, dim=1)
                original_pred_mean = mu[torch.arange(len(mu)), best_component]
                original_pred_var = torch.exp(log_var[torch.arange(len(log_var)), best_component])
            elif args.decoder_type == 'nll':
                original_pred_mean = original_pred[:, 0]  # Extract mean
                original_pred_var = torch.exp(original_pred[:, 1])  # Extract variance from log_var
            else:
                original_pred_mean = original_pred
                original_pred_var = torch.zeros_like(original_pred)
            
            # Get unique conditions in this batch
            unique_conditions = torch.unique(query_c)

            for i in range(len(original_pred)):
                patient_id = subject_ids[i]
                lab_code_val = lab_code[i].item()
                original_condition = query_c[i].item()
                original_prediction = original_pred_mean[i].item()
                original_variance = original_pred_var[i].item()
                original_target = y[i].item()
                
                # Generate counterfactuals for each possible condition
                for cf_condition in unique_conditions:
                    cf_condition_val = cf_condition.item()
                    if cf_condition_val != original_condition:
                        # Create counterfactual query
                        cf_query_c = cf_condition.unsqueeze(0).expand(query_c.size(0), -1)
                        cf_query_c[i] = cf_condition  # Set specific patient's condition
                        
                        # Make counterfactual prediction
                        cf_pred = model(x, t, c, sex, lab_code, query_t, cf_query_c, pad_mask)
                        if args.decoder_type == 'mdn':
                            pi, mu, log_var = cf_pred
                            best_component = torch.argmax(pi[i], dim=0)
                            cf_prediction = mu[i, best_component].item()
                            cf_variance = torch.exp(log_var[i, best_component]).item()
                        elif args.decoder_type == 'nll':
                            cf_prediction, cf_variance = cf_pred[i, 0].item(), cf_pred[i, 1].item()  # Extract mean from (mean, log_var)
                        else:
                            cf_prediction, cf_variance = cf_pred[i].item(), 0.0
                        
                        counterfactual_data.append({
                            'patient_id': patient_id,
                            'test_name': lab_code_val,
                            'original_condition': original_condition,
                            'counterfactual_condition': cf_condition_val,
                            'original_prediction': original_prediction,
                            'original_variance': original_variance,
                            'counterfactual_prediction': cf_prediction,
                            'original_target': original_target,
                            'counterfactual_variance': cf_variance,
                            'prediction_difference': cf_prediction - original_prediction
                        })
    
    # Save counterfactual results as CSV
    cf_df = pd.DataFrame(counterfactual_data)
    cf_df.to_csv(f"{save_dir}/counterfactual_predictions.csv", index=False)
    
    print(f"Generated counterfactual predictions for {len(cf_df)} patient-condition pairs")
    return cf_df

def sample_patients(df, num_patients, min_data_points=3):
    # First filter by minimum data points
    patient_counts = df['subject_id'].value_counts()
    valid_patients = patient_counts[patient_counts >= min_data_points].index
    df_filtered = df[df['subject_id'].isin(valid_patients)]
    
    print(f"Found {len(valid_patients)} patients with {min_data_points}+ data points")
    
    # Then sample if specified
    if num_patients:
        unique_patients = df_filtered['subject_id'].unique()
        sampled_patients = np.random.choice(unique_patients, size=min(num_patients, len(unique_patients)), replace=False)
        df_filtered = df_filtered[df_filtered['subject_id'].isin(sampled_patients)]
        print(f"Sampled {len(sampled_patients)} patients from {len(unique_patients)} valid patients")
    
    return df_filtered

# -------------------------
# Training loop
# -------------------------
def train(args):
    # Load data
    df = pd.read_csv(args.data_path).query("test_name == 'HGB'")
    
    # Filter and sample patients
    df = sample_patients(df, args.sample_patients, min_data_points=3)
    
    train_loader, val_loader, test_loader = create_dataloaders(df, batch_size=args.batch_size)
    # Get actual dimensions from data
    sample_batch = next(iter(train_loader))
    num_lab_codes = len(df['test_name'].unique())

    model = TimeConditionedTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        num_lab_codes=num_lab_codes,
        decoder_type=args.decoder_type,
        **args.decoder_kwargs
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    loss_fn = get_loss_fn(args.decoder_type, **args.decoder_kwargs)

    # Training
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for x, t, c, sex, lab_code, query_t, query_c, y, pad_mask, subject_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, t, c = x.to(device), t.to(device), c.to(device)
            sex, lab_code = sex.to(device), lab_code.to(device)
            query_t, query_c = query_t.to(device), query_c.to(device)
            y = y.to(device)
            pad_mask = pad_mask.to(device)

            optimizer.zero_grad()
            pred = model(x, t, c, sex, lab_code, query_t, query_c, pad_mask)

            # Handle different decoder outputs
            if args.decoder_type == 'mdn':
                pi, mu, log_var = pred
                loss = loss_fn(pi, mu, log_var, y)
            elif args.decoder_type == 'diffusion':
                # For diffusion, we need to add noise during training
                noise = torch.randn_like(y)
                timestep = torch.randint(0, 1000, (y.shape[0],), device=y.device)
                loss = loss_fn(pred, noise)
            else:
                loss = loss_fn(pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # Remove scheduler.step() from here

            running_loss += loss.item() * x.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, loss_fn, device, args.decoder_type)
        scheduler.step(val_loss)  # Only call once per epoch

        print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), f"models/model_checkpoint.pt")
    
    # Generate and save predictions
    print("\nGenerating predictions...")
    train_results = save_predictions(model, train_loader, "train", device, decoder_type=args.decoder_type)
    val_results = save_predictions(model, val_loader, "val", device, decoder_type=args.decoder_type)
    test_results = save_predictions(model, test_loader, "test", device, decoder_type=args.decoder_type)
    
    # Optional: Only generate counterfactuals if requested
    if args.generate_counterfactuals:
        print("\nGenerating counterfactual predictions...")
        counterfactual_results = generate_counterfactuals(model, test_loader, device)
    
    # Save summary as CSV
    summary_df = pd.DataFrame([{
        'split': 'train',
        'mse': train_results['mse'],
        'mae': train_results['mae']
    }, {
        'split': 'val', 
        'mse': val_results['mse'],
        'mae': val_results['mae']
    }, {
        'split': 'test',
        'mse': test_results['mse'], 
        'mae': test_results['mae']
    }])
    
    summary_df.to_csv("predictions/training_summary.csv", index=False)
    
    print(f"\nAll results saved to 'predictions/' directory as CSV files")
    return summary_df

# -------------------------
# CLI entrypoint
# -------------------------
def main(args):
    train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../../data/processed/lab_measurements.csv")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d_model', type=int, default=64)  # Reduce from 128
    parser.add_argument('--nhead', type=int, default=2)     # Reduce from 4
    parser.add_argument('--num_layers', type=int, default=2) # Reduce from 4
    parser.add_argument('--decoder_type', type=str, 
                       choices=['standard', 'nll', 'mdn', 'quantile', 'diffusion'], 
                       default='mdn', help='Type of decoder head')
    parser.add_argument('--sample_patients', type=int, default=None)  # Start with 100
    parser.add_argument('--generate_counterfactuals', type=bool, default=True)
    
    # Decoder-specific arguments
    parser.add_argument('--num_components', type=int, default=5, help='Number of MDN components')
    parser.add_argument('--num_quantiles', type=int, default=9, help='Number of quantiles for quantile regression')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    args = parser.parse_args()
    
    # Set up decoder-specific kwargs
    args.decoder_kwargs = {}
    if args.decoder_type == 'mdn':
        args.decoder_kwargs['num_components'] = args.num_components
    elif args.decoder_type == 'quantile':
        args.decoder_kwargs['num_quantiles'] = args.num_quantiles
    elif args.decoder_type == 'diffusion':
        args.decoder_kwargs['num_timesteps'] = args.num_timesteps

    main(args)
    print("Program finished executing")
