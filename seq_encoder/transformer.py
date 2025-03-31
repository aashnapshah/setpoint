import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import math
from datetime import datetime, timedelta
import random

#############################################################################################################
class CBCTransformer(nn.Module):
    def __init__(self, 
                 input_dim=3,           ### Input features (RBC value, time gap, demographics)
                 model_dim=64,          ### Hidden dimension
                 num_heads=4,           ### Number of attention heads
                 num_layers=3,          ### Number of transformer layers
                 dropout=0.5
                ):
        super(CBCTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.model_dim = model_dim
        
        # Transformer backbone
        self.embedding = nn.Linear(input_dim, model_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, 
                                                       nhead=num_heads,
                                                       dropout=dropout,
                                                       batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, 
                                               num_layers=num_layers)
        
        # Distribution parameters
        self.mean_fc = nn.Linear(model_dim, 1)
        self.variance_fc = nn.Linear(model_dim, 1)
        
        # Time encoding
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        for layer in [self.mean_fc, self.variance_fc]:
            nn.init.uniform_(layer.weight, -initrange, initrange)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # Embed input
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.dropout(x + self.pos_encoder(x))
        x = self.layer_norm(x)
        
        # Transform sequence
        x = self.transformer(x)
        
        # Get distribution parameters
        mean = self.mean_fc(x)
        # Ensure variance is positive and reasonable for RBC values
        variance = torch.exp(self.variance_fc(x)) * 0.1
        
        return mean, variance

#############################################################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

#############################################################################################################
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

#############################################################################################################
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

#############################################################################################################
def train_model(model, data, sequence_lengths, epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.GaussianNLLLoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0
        
        for i in range(len(data)):
            seq_len = sequence_lengths[i]
            input_seq = data[i, :seq_len-1].unsqueeze(0)
            target = data[i, seq_len-1, 0].unsqueeze(0)
            
            mean, variance = model(input_seq)
            loss = loss_fn(mean[:, -1], target, variance[:, -1])
            total_loss += loss
        
        avg_loss = total_loss / len(data)
        avg_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Loss: {avg_loss.item()}")
            
#############################################################################################################
if __name__ == "__main__":
    print("Testing RBC Transformer Model\n")
    
    # Generate data
    print("Generating synthetic RBC data...")
    data, sequence_lengths = generate_realistic_rbc_data(num_samples=100, max_sequence_length=10)
    
    print(data)
    print(f"Generated {len(data)} sequences")
    print(f"Example sequence shape: {data[0].shape}")
    print(f"Example sequence:\n RBC values: {data[0, :5, 0]}\n Time gaps: {data[0, :5, 1]}\n Demographics: {data[0, :5, 2]}\n")
    
    # Initialize model
    print("Initializing model...")
    model = CBCTransformer()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("\nTraining model...")
    train_model(model, data, sequence_lengths, epochs=100)
    
    # Test predictions
    print("\nTesting predictions...")
    test_sequences = [
        data[0, :5],  # First patient, 5 measurements
        data[1, :7],  # Second patient, 7 measurements
        data[2, :6]   # Third patient, 6 measurements
    ]
    
    for i, test_seq in enumerate(test_sequences):
        print(f"\nPatient {i+1}:")
        print(f"History (RBC values): {test_seq[:, 0].tolist()}")
        print(f"Gender: {'Female' if test_seq[0, 2] == 1 else 'Male'}")
        
        # Calculate reference ranges at different confidence levels
        for confidence in [0.68, 0.95]:
            lower, mean, upper = calculate_reference_range(model, test_seq, confidence)
            print(f"{confidence*100}% Reference Range:")
            print(f"  {lower:.2f} - {upper:.2f} (mean: {mean:.2f})")
        
        # Make next prediction
        with torch.no_grad():
            mean, variance = model(test_seq.unsqueeze(0))
            pred_mean = mean[0, -1, 0].item()
            pred_std = torch.sqrt(variance[0, -1, 0]).item()
            print(f"Next predicted RBC: {pred_mean:.2f} ± {pred_std:.2f}")

    print("\nTesting complete!")