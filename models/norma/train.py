import torch
import torch.nn as nn
from tqdm import tqdm
from model import DualDecoder, ConditionalDecoder, TimeConditionedTransformer
from data import create_dataloaders
from loss import DualModeLoss, SingleModeLoss
from data_utils import compute_comprehensive_metrics, create_metrics_summary
from plotting_utils import create_all_plots
import pandas as pd
import argparse
import os
from datetime import datetime
import wandb
import numpy as np
from datetime import datetime
import json

def load_data(filename, num_patients, min_data_points=5):
    df = pd.read_csv(filename)
    subject_test_counts = df.groupby(['subject_id', 'test_name']).size()
    valid_subject_tests = subject_test_counts[subject_test_counts >= min_data_points]
    df_filtered = df.merge(valid_subject_tests.reset_index(), on=['subject_id', 'test_name'], how='inner').drop(columns=[0])
    
    if num_patients:
        unique_patients = df_filtered['subject_id'].unique()
        sampled_patients = np.random.choice(unique_patients, size=min(num_patients, len(unique_patients)), replace=False)
        df_filtered = df_filtered[df_filtered['subject_id'].isin(sampled_patients)]
    
    return df_filtered

def save_model(model, optimizer, epoch, val_loss, args, filename_prefix="model"):
    os.makedirs("models", exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': float(val_loss)
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = 'dual_mode' if isinstance(model, DualDecoder) else 'single_mode'
    filename = f"models/{filename_prefix}_{model_type}_{timestamp}.pt"
    torch.save(save_dict, filename)
    
    if args.wandb and filename_prefix == "best_model":
        artifact = wandb.Artifact(
            name=f"{filename_prefix}_{model_type}",
            type="model",
            description=f"Best {model_type} model with val_loss: {val_loss:.4f}"
        )
        artifact.add_file(filename)
        wandb.log_artifact(artifact)
    
    return filename

def train(args):
    if args.wandb:
        wandb.init(project="setpoint-norma", config=vars(args))

    df = load_data(args.data_path, args.sample_patients)
    train_loader, val_loader, test_loader = create_dataloaders(df, args.batch_size)
    num_lab_codes = len(df['test_name'].unique())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_classes = {
        'dual_mode': (DualDecoder, DualModeLoss),
        'single_mode': (ConditionalDecoder, SingleModeLoss),
        'time_conditioned': (TimeConditionedTransformer, nn.MSELoss)
    }
    
    ModelClass, LossClass = model_classes[args.model_type]
    model = ModelClass(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        num_lab_codes=num_lab_codes
    ).to(device)
    
    loss_fn = LossClass(lambda_align=args.lambda_align, adaptive_weight=args.adaptive_weight) if args.model_type in ['dual_mode', 'single_mode'] else LossClass()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss, train_forecast, train_align = run_epoch(model, train_loader, loss_fn, optimizer, device, train_mode=True)
        val_loss, val_metrics, _, _ = compute_comprehensive_metrics(model, val_loader, loss_fn, device, "val")
        
        if args.wandb:
            log_dict = {
                'train/loss': train_loss,
                'val/loss': val_loss,
                'epoch': epoch,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'val/r2': val_metrics['r2'],
                'val/mae': val_metrics['mae'], 
                'val/mse': val_metrics['mse']
            }

            if args.model_type in ['dual_mode', 'single_mode']:
                log_dict.update({
                    'train/forecast_loss': train_forecast,
                    'train/align_loss': train_align,
                    'val/forecast_loss': val_metrics.get('forecast_loss', 0),
                    'val/align_loss': val_metrics.get('align_loss', 0)
                })

            wandb.log(log_dict)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, optimizer, epoch, val_loss, args, "best_model")

            scheduler.step(val_loss)

    save_model(model, optimizer, args.epochs-1, val_loss, args, "final_model")

    # Final evaluation and metrics summary
    print("\n=== Final Evaluation ===")
    train_loss, train_overall, train_per_test, train_df = compute_comprehensive_metrics(
        model, train_loader, loss_fn, device, "train"
    )
    val_loss, val_overall, val_per_test, val_df = compute_comprehensive_metrics(
        model, val_loader, loss_fn, device, "val"
    )
    test_loss, test_overall, test_per_test, test_df = compute_comprehensive_metrics(
        model, test_loader, loss_fn, device, "test"
    )
    
    # Print final results
    print(f"Train | Loss: {train_loss:.4f} | R2: {train_overall['r2']:.4f} | MAE: {train_overall['mae']:.4f}")
    print(f"Val   | Loss: {val_loss:.4f} | R2: {val_overall['r2']:.4f} | MAE: {val_overall['mae']:.4f}")
    print(f"Test  | Loss: {test_loss:.4f} | R2: {test_overall['r2']:.4f} | MAE: {test_overall['mae']:.4f}")


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.model_type}_{timestamp}"
    run_dir = f"predictions/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/hyperparameters.json", "w") as f:
        json.dump(vars(args), f)
    print(f"Creating directory: {run_dir}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Saving predictions - Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows")
    train_df.to_csv(f"{run_dir}/train_predictions_{run_id}.csv", index=False)
    val_df.to_csv(f"{run_dir}/val_predictions_{run_id}.csv", index=False)  
    test_df.to_csv(f"{run_dir}/test_predictions_{run_id}.csv", index=False)
    print(f"Saved individual prediction files to {run_dir}")
    
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df.to_csv(f"{run_dir}/all_predictions_{run_id}.csv", index=False)
    
    metrics_summary = create_metrics_summary(
        (train_overall, train_per_test),
        (val_overall, val_per_test), 
        (test_overall, test_per_test)
    )
    metrics_summary['run_id'] = run_id
    metrics_summary['timestamp'] = datetime.now().isoformat()
    metrics_summary.to_csv(f"{run_dir}/metrics_summary_{run_id}.csv", index=False)
    
    print("Creating visualization plots...")
    
    plots_dir = f"{run_dir}/plots"
    figures = create_all_plots(
        metrics_summary=metrics_summary,
        test_predictions=test_df,
        save_dir=plots_dir,
        wandb_module=wandb if args.wandb else None
    )
    print(f"Plots saved to {plots_dir}")

    if args.wandb:
        # Log final summary metrics table with only the 7 metrics columns
        metrics_summary_table = wandb.Table(columns=[
            "split", "loss", "r2", "mae", "mse", "forecast_loss", "align_loss"
        ])
        
        for split, overall_metrics in [
            ("train", train_overall), 
            ("val", val_overall), 
            ("test", test_overall)
        ]:
            metrics_summary_table.add_data(
                split,
                overall_metrics.get('loss', 0),
                overall_metrics.get('r2', 0),
                overall_metrics.get('mae', 0),
                overall_metrics.get('mse', 0),
                overall_metrics.get('forecast_loss', 0),
                overall_metrics.get('align_loss', 0)
            )
        
        wandb.log({"metrics_summary": metrics_summary_table})
        wandb.finish()

def run_epoch(model, loader, loss_fn, optimizer, device, train_mode=True):
    model.train() if train_mode else model.eval()
    total_loss = total_forecast = total_align = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="Train" if train_mode else "Valid"):
        x, t, c, sex, lab_code, query_t, query_c, y, ref_mu, ref_var, pad_mask, _ = batch

        if train_mode:
            optimizer.zero_grad()

        if isinstance(model, DualDecoder):
            P_healthy, P_unhealthy = model(x, t, sex, lab_code, query_t, pad_mask)
            condition = query_c.squeeze(1)
            ref_sigma = torch.sqrt(ref_var)
            loss, forecast_loss, align_loss = loss_fn(P_healthy, P_unhealthy, y, condition, ref_mu, ref_sigma)
        elif isinstance(model, ConditionalDecoder):
            mu, log_var = model(x, t, c, sex, lab_code, query_t, query_c, pad_mask)
            condition = query_c.squeeze(1).float()
            ref_sigma = torch.sqrt(ref_var)
            loss, forecast_loss, align_loss = loss_fn(mu, log_var, y, condition, ref_mu, ref_sigma)
        else:
            output = model(x, t, c, sex, lab_code, query_t, query_c, pad_mask)
            loss = loss_fn(output, y)
            forecast_loss = align_loss = torch.tensor(0.0, device=device)

        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_forecast += forecast_loss.item()
        total_align += align_loss.item()
        num_batches += 1

    return total_loss/num_batches, total_forecast/num_batches, total_align/num_batches

def main(args):
    train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../../data/processed/lab_measurements.csv")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--sample_patients', type=int, default=None)
    parser.add_argument('--model_type', type=str, choices=['dual_mode', 'single_mode'], default='single_mode')
    parser.add_argument('--lambda_align', type=float, default=0.01)
    parser.add_argument('--adaptive_weight', type=bool, default=True)
    parser.add_argument('--wandb', type=bool, default=True)
    args = parser.parse_args()
    args.decoder_kwargs = {}
    main(args)