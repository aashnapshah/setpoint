import torch
import torch.nn as nn
import math

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