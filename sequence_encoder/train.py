import time
import math
import copy
import torch
from torch.optim import Adam
import wandb
import argparse
from model_utils import parse_args, set_seed, get_hparams, load_split_data, loop_batch
from model import CBCTransformer

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda": print(torch.cuda.get_device_name(0))


def train(epochs, batch_sz, split, train_loader, val_loader, model, temporal_len, optimizer, loss_fn, save_model_f, save_preds=False, time_skip=False):
    metrics = {
        "train_loss": [], "val_loss": []
        # "train_r2": [], "train_mae": [], "train_mape": [],
        # "val_r2": [], "val_mae": [], "val_mape": []
    }
    best_model_dict = {"best_model": None, "best_val": math.inf}
    model.train()
    
    for epoch in range(epochs):
        start = time.time()
        
        # Train Loop
        overall_train_loss, train_metrics, model, all_predictions = loop_batch(
            "train", batch_sz, split, train_loader, model, train_loader,
            temporal_len, loss_fn, optimizer, save_preds=save_preds,
            time_skip=time_skip, device=device, wandb=wandb
        )
        
        # Validation loop
        overall_val_loss, val_metrics, model, all_predictions = loop_batch(
            "val", batch_sz, split, val_loader, model, train_loader,
            temporal_len, loss_fn, save_preds=save_preds,
            time_skip=time_skip, device=device, wandb=wandb
        )
        
        # Save model
        if overall_val_loss < best_model_dict["best_val"]:
            with open(save_model_f + f"model_ckpt_split={split}_temporal={temporal_len}_epoch={epoch}_valr2={val_metrics['r2']:.2f}.pth", 'wb') as f:
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }, f)
            best_model_dict["best_model"] = copy.deepcopy(model)
            best_model_dict["best_val"] = overall_val_loss
        
        # Record metrics
        end = time.time()
        train_stats = {
            "Epoch": epoch + 1,
            "Total Train Loss": overall_train_loss
            # "Total Train R2": train_metrics["r2"],
            # "Total Train MAE": train_metrics["mae"],
            # "Total Train MAPE": train_metrics["mape"]
        }
        
        val_stats = {
            "Total Val Loss": overall_val_loss,
            # "Total Val R2": val_metrics["r2"],
            # "Total Val MAE": val_metrics["mae"],
            # "Total Val MAPE": val_metrics["mape"],
            "Time Lapsed": round(end - start, 2)
        }
        
        if epoch % 10 == 0:
            print(train_stats)
            print(val_stats)
        #wandb.log({**train_stats, **val_stats})
        
        # Store metrics history
        metrics["train_loss"].append(overall_train_loss)
        metrics["val_loss"].append(overall_val_loss)
        # metrics["train_r2"].append(train_metrics["r2"])
        # metrics["train_mae"].append(train_metrics["mae"])
        # metrics["train_mape"].append(train_metrics["mape"])
        # metrics["val_r2"].append(val_metrics["r2"])
        # metrics["val_mae"].append(val_metrics["mae"])
        # metrics["val_mape"].append(val_metrics["mape"])
    
    print("Finish training!\n")
    return metrics, best_model_dict

def test(batch_sz, split, train_loader, test_loader, model, temporal_len, loss_fn, save_preds=False, time_skip=False):
    model.to(device)
    model.eval()
    start = time.time()
    
    overall_test_loss, test_metrics, model, all_test_preds = loop_batch(
        "test", batch_sz, split, test_loader, model, train_loader,
        temporal_len, loss_fn, save_preds=save_preds,
        time_skip=time_skip, device=device, wandb=wandb
    )
    
    end = time.time()
    test_stats = {
        "Total Test Loss": overall_test_loss,
        # "Total Test R2": test_metrics["r2"],
        # "Total Test MAE": test_metrics["mae"],
        # "Total Test MAPE": test_metrics["mape"],
        "Time Lapsed": round(end - start, 2)
    }
    print(test_stats)
    #wandb.log(test_stats)
    
    return overall_test_loss, test_metrics, all_test_preds

def main():
    # Parse args and hparams
    args = parse_args()
    hparams = get_hparams(args)

    # Set seed
    set_seed(args.seed)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = load_split_data(args.data_dir, hparams)
    
    print("Train:", train_loader)
    print("Val:", val_loader)
    print("Test:", test_loader)
    
    # Initialize model
    print("Initializing model...")
    print(hparams)
    #wandb.init(config=hparams, project="cbc-transformer", entity="your-entity")
    #hparams = wandb.config
    
    model = CBCTransformer(
        input_dim=hparams["input_dim"],
        model_dim=hparams["model_dim"],
        num_heads=hparams["num_heads"],
        num_layers=hparams["num_layers"],
        dropout=hparams["dropout"]
    )
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=hparams["lr"])
    loss_fn = torch.nn.GaussianNLLLoss()
    #wandb.watch(model)
    
    # Train
    print("Starting training...")
    metrics, best_model_dict = train(
        hparams["epochs"], hparams["batch_size"], hparams["split"],
        train_loader, val_loader, model, hparams["temporal_len"],
        optimizer, loss_fn, args.output_dir + args.save_prefix,
        save_preds=args.save_preds, time_skip=args.time_skip
    )
    
    # Test
    print("Starting inference...")
    test_loss, test_metrics, test_preds = test(
        hparams["batch_size"], hparams["split"],
        train_loader, test_loader, best_model_dict["best_model"],
        hparams["temporal_len"], loss_fn,
        save_preds=args.save_preds, time_skip=args.time_skip
    )
    
    if args.save_preds:
        save_model_preds(test_preds, args.output_dir + args.save_prefix)
        
    print("Finished.")

if __name__ == "__main__":
    main()