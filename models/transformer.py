import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt

# 1. Model Definition
class SetpointTrajectoryUncertaintyModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super(SetpointTrajectoryUncertaintyModel, self).__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.trajectory_head = nn.Linear(hidden_dim, 1)
        self.setpoint_mu = nn.Linear(hidden_dim, 1)
        self.setpoint_logvar = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_dist=False):
        rnn_out, h_n = self.rnn(x)
        last_hidden = h_n.squeeze(0)
        trajectory_preds = self.trajectory_head(rnn_out)
        mu = self.setpoint_mu(last_hidden)
        logvar = self.setpoint_logvar(last_hidden)
        sigma = torch.exp(0.5 * logvar)

        if return_dist:
            return trajectory_preds, mu, sigma
        else:
            eps = torch.randn_like(sigma)
            sampled_setpoint = mu + eps * sigma
            return trajectory_preds, sampled_setpoint

# 2. Synthetic Dataset
class SyntheticLabDataset(Dataset):
    def __init__(self, num_patients=1000, seq_len=15, input_dim=10):
        super().__init__()
        self.data = []
        for _ in range(num_patients):
            base_value = random.uniform(0.5, 1.5)
            time_series = np.random.randn(seq_len, input_dim - 1) * 0.1
            time_deltas = np.linspace(0, 1, seq_len).reshape(-1, 1)
            patient_features = np.hstack([time_series, time_deltas])
            true_curve = base_value + np.sin(time_deltas * 3 * np.pi) * 0.1
            self.data.append((torch.tensor(patient_features, dtype=torch.float32),
                              torch.tensor(true_curve, dtype=torch.float32),
                              torch.tensor([[base_value]], dtype=torch.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data[idx]

# 3. Loss Function
def custom_loss(trajectory_pred, trajectory_true, mu, sigma, setpoint_true):
    mse_loss = F.mse_loss(trajectory_pred, trajectory_true)
    # NLL loss for Gaussian (setpoint)
    nll_loss = torch.mean(torch.log(sigma**2 + 1e-6) + ((setpoint_true - mu)**2) / (sigma**2 + 1e-6))
    return mse_loss + 0.5 * nll_loss

# 4. Training Loop
def train_model(model, dataloader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x, y, setpoint = batch
            optimizer.zero_grad()
            traj_pred, mu, sigma = model(x, return_dist=True)
            loss = custom_loss(traj_pred, y, mu, sigma, setpoint)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# 5. Run Training
input_dim = 10
hidden_dim = 64
dataset = SyntheticLabDataset(num_patients=500, seq_len=15, input_dim=input_dim)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = SetpointTrajectoryUncertaintyModel(input_dim=input_dim, hidden_dim=hidden_dim)

train_model(model, dataloader, epochs=10)

# 6. Evaluate One Sample
model.eval()
sample_x, sample_y, sample_setpoint = dataset[0]
sample_x = sample_x.unsqueeze(0)
with torch.no_grad():
    pred_traj, mu, sigma = model(sample_x, return_dist=True)

# 7. Visualize
plt.plot(sample_y.numpy(), label="True Values")
plt.plot(pred_traj.squeeze().numpy(), label="Predicted Trajectory")
plt.axhline(mu.item(), color='r', linestyle='--', label=f"Setpoint μ = {mu.item():.2f}")
plt.fill_between(np.arange(len(sample_y)),
                 mu.item() - sigma.item(),
                 mu.item() + sigma.item(),
                 color='r', alpha=0.2, label='Setpoint ±σ')
plt.title("Lab Value Trajectory and Setpoint")
plt.legend()
plt.xlabel("Time Step")
plt.ylabel("Lab Value")
plt.grid(True)
plt.tight_layout()
plt.show()

