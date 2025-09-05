import torch
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, List, Tuple, Any, Optional

TEST_VOCAB = {
    'WBC': 0, 'HCT': 1, 'A1C': 2, 'HGB': 3, 'PLT': 4, 'MCH': 5, 'MCHC': 6, 'MCV': 7, 
    'RDW': 8, 'RBC': 9, 'CA': 10, 'CO2': 11, 'CL': 12, 'CRE': 13, 'GLU': 14, 'K': 15, 
    'BUN': 16, 'ALT': 17, 'ALB': 18, 'AST': 19, 'TBIL': 20, 'TP': 21, 'ALP': 22, 
    'PT': 23, 'DBIL': 24, 'HDL': 25, 'TC': 26, 'CRP': 27, 'MPV': 28, 'LDL': 29, 
    'GGT': 30, 'LDH': 31
}
CODE_TO_TEST_NAME = {i: test_name for test_name, i in TEST_VOCAB.items()}

def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    # Move tensors to CPU before converting to numpy if they're on GPU
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
        
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    return {
        'mae': mae,
        'mse': mse,
        'r2': r2
    }

def compute_per_test_metrics(results_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    per_test_metrics = {}
    
    for test_name in results_df['test_name'].unique():
        test_data = results_df[results_df['test_name'] == test_name]
        predictions = test_data['prediction'].values
        targets = test_data['target'].values
        
        metrics = compute_metrics(predictions, targets)
        metrics['num_samples'] = len(predictions)
        
        per_test_metrics[str(test_name)] = metrics
    
    return per_test_metrics

def compute_comprehensive_metrics(model, loader, loss_fn, device, split_name: str = "eval") -> Tuple[float, Dict[str, float], Dict[str, Dict[str, float]], pd.DataFrame]:
    from model import DualDecoder, ConditionalDecoder, TimeConditionedTransformer
    
    model.eval()
    total_loss = 0.0
    total_forecast_loss = 0.0
    total_align_loss = 0.0
    count = 0
    records = []
    
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()
    
    with torch.no_grad():
        for batch in loader:
            x, t, c, sex, lab_code, query_t, query_c, y, ref_mu, ref_var, pad_mask, subject_ids = batch
            x, t, c, sex, lab_code, query_t, query_c, y, ref_mu, ref_var, pad_mask = [
                b.to(device) for b in [x, t, c, sex, lab_code, query_t, query_c, y, ref_mu, ref_var, pad_mask]
            ]
            
            batch_size = x.size(0)
            count += batch_size
            
            if isinstance(model, DualDecoder):
                P_healthy, P_unhealthy = model(x, t, sex, lab_code, query_t, pad_mask)
                condition = query_c.squeeze(1)
                ref_sigma = torch.sqrt(ref_var)
                loss, forecast_loss, align_loss = loss_fn(
                    P_healthy, P_unhealthy, y, condition, ref_mu, ref_sigma
                )
                
                mu_h, log_var_h = P_healthy
                mu_u, log_var_u = P_unhealthy
                condition_float = condition.float()
                pred_mean = torch.where(condition_float == 0, mu_h, mu_u)
                pred_var = torch.where(condition_float == 0, torch.exp(log_var_h), torch.exp(log_var_u))
                
            elif isinstance(model, ConditionalDecoder):
                mu, log_var = model(x, t, c, sex, lab_code, query_t, query_c, pad_mask)
                condition = query_c.squeeze(1).float()
                ref_sigma = torch.sqrt(ref_var)
                loss, forecast_loss, align_loss = loss_fn(
                    mu, log_var, y, condition, ref_mu, ref_sigma
                )
                
                pred_mean = mu
                pred_var = torch.exp(log_var)
                
            else:
                output = model(x, t, c, sex, lab_code, query_t, query_c, pad_mask)
                loss = loss_fn(output, y)
                forecast_loss = align_loss = torch.tensor(0.0, device=device)
                
                pred_mean = output
                pred_var = torch.zeros_like(output)
            
            total_loss += loss.item() * batch_size
            total_forecast_loss += forecast_loss.item() * batch_size
            total_align_loss += align_loss.item() * batch_size
            
            for i in range(batch_size):
                test_code = lab_code[i].item()
                test_name = CODE_TO_TEST_NAME.get(test_code, f"TEST_{test_code}")
                
                record = {
                    'subject_id': subject_ids[i].item(),
                    'test_name': test_name,
                    'prediction': pred_mean[i].item(),
                    'variance': pred_var[i].item() if pred_var is not None else 0.0,
                    'target': y[i].item(),
                    'condition': query_c[i].item()
                }
                
                if isinstance(model, DualDecoder):
                    mu_h, log_var_h = P_healthy
                    mu_u, log_var_u = P_unhealthy
                    record.update({
                        'pred_healthy': mu_h[i].cpu().item(),
                        'pred_unhealthy': mu_u[i].cpu().item(),
                        'pred_healthy_var': torch.exp(log_var_h)[i].cpu().item(),
                        'pred_unhealthy_var': torch.exp(log_var_u)[i].cpu().item(),
                    })
                elif isinstance(model, ConditionalDecoder):
                    with torch.no_grad():
                        x_single = x[i:i+1]
                        t_single = t[i:i+1]
                        c_single = c[i:i+1]
                        sex_single = sex[i:i+1]
                        lab_code_single = lab_code[i:i+1]
                        query_t_single = query_t[i:i+1]
                        pad_mask_single = pad_mask[i:i+1] if pad_mask is not None else None
                        
                        query_c_healthy = torch.ones_like(query_c[i:i+1]).to(device)
                        mu_h, log_var_h = model(x_single, t_single, c_single, sex_single, 
                                               lab_code_single, query_t_single, query_c_healthy, pad_mask_single)
                        
                        query_c_unhealthy = torch.zeros_like(query_c[i:i+1]).to(device)
                        mu_u, log_var_u = model(x_single, t_single, c_single, sex_single,
                                               lab_code_single, query_t_single, query_c_unhealthy, pad_mask_single)
                    
                    record.update({
                        'pred_healthy': mu_h[0].cpu().item(),
                        'pred_unhealthy': mu_u[0].cpu().item(),
                        'pred_healthy_var': torch.exp(log_var_h)[0].cpu().item(),
                        'pred_unhealthy_var': torch.exp(log_var_u)[0].cpu().item(),
                    })
                else:
                    record.update({
                        'pred_healthy': None,
                        'pred_unhealthy': None,
                        'pred_healthy_var': None,
                        'pred_unhealthy_var': None,
                    })
                
                records.append(record)
    
    results_df = pd.DataFrame(records)
    
    predictions = results_df['prediction'].values
    targets = results_df['target'].values
    overall_metrics = compute_metrics(predictions, targets)
    overall_metrics['loss'] = total_loss / count
    
    if isinstance(model, (DualDecoder, ConditionalDecoder)):
        overall_metrics['forecast_loss'] = total_forecast_loss / count
        overall_metrics['align_loss'] = total_align_loss / count
    
    per_test_metrics = compute_per_test_metrics(results_df)
    
    return overall_metrics['loss'], overall_metrics, per_test_metrics, results_df

def log_metrics_to_wandb(wandb_module, epoch: int, split: str, overall_metrics: Dict[str, float], per_test_metrics: Dict[str, Dict[str, float]], lr: float = None):
    log_dict = {'epoch': epoch}
    
    for metric_name, value in overall_metrics.items():
        log_dict[f'{split}/{metric_name}'] = value
    
    for test_name, metrics in per_test_metrics.items():
        for metric_name, value in metrics.items():
            if metric_name != 'num_samples':
                log_dict[f'{split}/{test_name}/{metric_name}'] = value
    
    if lr is not None:
        log_dict['learning_rate'] = lr
    
    wandb_module.log(log_dict)

def create_metrics_summary(train_metrics: Dict[str, Any], val_metrics: Dict[str, Any], test_metrics: Dict[str, Any]) -> pd.DataFrame:
    summary_data = []
    
    for split, (overall, per_test) in [
        ('train', train_metrics), 
        ('val', val_metrics), 
        ('test', test_metrics)
    ]:
        row = {'split': split, 'test_name': 'overall'}
        row.update(overall)
        summary_data.append(row)
        
        for test_name, metrics in per_test.items():
            row = {'split': split, 'test_name': test_name}
            row.update(metrics)
            summary_data.append(row)
    
    return pd.DataFrame(summary_data)
