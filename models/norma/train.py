import torch
import torch.nn as nn
from dataset import build_datasets
from model import TransformerEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda": 
    print(torch.cuda.get_device_name(0))

def train(model, train_loader, val_loader, epochs=10, learning_rate=0.001, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    metrics = {
        "train_loss": [], "val_loss": [],
        "train_rmse": [], "val_rmse": []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for x, time_dict, target in train_loader:
            x = x.to(device)
            target = target.to(device)
            for k in time_dict:
                time_dict[k] = time_dict[k].to(device)
            
            # Create padding mask
            src_key_padding_mask = (x.squeeze(-1) == 0).to(device)
            
            optimizer.zero_grad()
            pred = model(x, src_key_padding_mask, dates=time_dict, device=device)
            loss = criterion(pred.squeeze(), target)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for x, time_dict, target in val_loader:
                x = x.to(device)
                target = target.to(device)
                for k in time_dict:
                    time_dict[k] = time_dict[k].to(device)
                
                src_key_padding_mask = (x.squeeze(-1) == 0).to(device)
                pred = model(x, src_key_padding_mask, dates=time_dict, device=device)
                loss = criterion(pred.squeeze(), target)
                val_losses.append(loss.item())
        
        # Record metrics
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_rmse"].append(torch.sqrt(torch.tensor(train_loss)).item())
        metrics["val_rmse"].append(torch.sqrt(torch.tensor(val_loss)).item())
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, RMSE: {metrics["train_rmse"][-1]:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, RMSE: {metrics["val_rmse"][-1]:.4f}')
    
    return metrics

def test_model(model, test_loader, device):
    """Test the trained model"""
    model.eval()
    test_losses = []
    predictions = []
    targets = []
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for x, time_dict, target in test_loader:
            x = x.to(device)
            target = target.to(device)
            for k in time_dict:
                time_dict[k] = time_dict[k].to(device)
            
            src_key_padding_mask = (x.squeeze(-1) == 0).to(device)
            pred = model(x, src_key_padding_mask, dates=time_dict, device=device)
            loss = criterion(pred.squeeze(), target)
            
            test_losses.append(loss.item())
            predictions.extend(pred.squeeze().cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    test_loss = sum(test_losses) / len(test_losses)
    test_rmse = torch.sqrt(torch.tensor(test_loss)).item()
    
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    
    return test_loss, test_rmse, predictions, targets

def main():
    # Build datasets
    data_path = '../../data/processed/lab_measurements.csv'
    train_loader, val_loader, test_loader, max_seq_len = build_datasets(data_path, sample_frac=0.5)

    # Print first 5 samples
    for i, (x, time_dict, target) in enumerate(train_loader):
        print(f"Sample {i+1}:")
        print(f"  x: {x.shape}")
        print(f"  time_dict: {time_dict}")
        print(f"  x: {x}")
        print(f"  target: {target}")
        if i == 4: break
    
    print(f"Dataset loaded with max sequence length: {max_seq_len}")
    print(f"Train loader length: {len(train_loader)}")
    print(f"Val loader length: {len(val_loader)}")
    print(f"Test loader length: {len(test_loader)}")
    
    # Create model
    model = TransformerEncoder(
        ntoken=max_seq_len,
        ninp=64,
        nhead=4,
        nhid=128,
        nlayers=2,
        nout=1,
        dropout=0.1,
        position="time"
    )
    
    # Train model
    print("\nStarting training...")
    metrics = train(model, train_loader, val_loader, epochs=10, learning_rate=0.001, device=device)
    
    # Test model
    test_loss, test_rmse, predictions, targets = test_model(model, test_loader, device)
    
    # Print sample predictions
    print(f"\nSample predictions vs targets:")
    for i in range(min(5, len(predictions))):
        print(f"  Sample {i}: Predicted {predictions[i]:.2f}, Actual {targets[i]:.2f}")
    
    return model, metrics

if __name__ == "__main__":
    model, metrics = main()
