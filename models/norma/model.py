import torch
import torch.nn as nn

class Time2Vec(nn.Module):
    """Time2Vec embedding module (linear + periodic parts)."""
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, d_model - 1)

    def forward(self, t):
        v_linear = self.linear(t)                        # (B, T, 1)
        v_periodic = torch.sin(self.periodic(t))         # (B, T, D-1)
        return torch.cat([v_linear, v_periodic], dim=-1) # (B, T, D)

class TransformerEncoder(nn.Module):
    """Shared base logic for transformer models."""
    def __init__(self, d_model, nhead, num_layers, num_lab_codes):
        super().__init__()
        self.value_embed = nn.Linear(1, d_model)
        self.sex_embed = nn.Embedding(2, d_model)
        self.lab_code_embed = nn.Embedding(num_lab_codes, d_model)
        self.time_embed = Time2Vec(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def generate_causal_mask(self, seq_len, device):
        """Generate lower triangular mask for autoregressive attention"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def encode_sequence(self, x, t, sex, lab_code, pad_mask=None, causal=True):
        B, T = x.shape[:2]
        sex_emb = self.sex_embed(sex).squeeze(1).unsqueeze(1).expand(B, T, -1)
        lab_emb = self.lab_code_embed(lab_code).squeeze(1).unsqueeze(1).expand(B, T, -1)

        encoder_input = self.value_embed(x) + self.time_embed(t) + sex_emb + lab_emb
        attn_mask = self.generate_causal_mask(T, x.device)
        
        encoded = self.encoder(encoder_input, 
                              mask=attn_mask,
                              src_key_padding_mask=pad_mask)

        # if pad_mask is not None:
        #     mask = (~pad_mask).float().unsqueeze(-1)
        #     return (encoded * mask).sum(dim=1) / mask.sum(dim=1)
        return encoded[:, -1] #.mean(dim=1)

class ConditionalDecoder(TransformerEncoder):
    """Predicts a single distribution conditioned on query time and condition."""
    def __init__(self, d_model=128, nhead=4, num_layers=4, num_lab_codes=2):
        super().__init__(d_model, nhead, num_layers, num_lab_codes)

        self.query_time_embed = Time2Vec(d_model)
        self.query_cond_embed = nn.Embedding(2, d_model)
        self.query_proj = nn.Linear(d_model * 2, d_model)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 2)
        )

    def process_query(self, query_t, query_c):
        q_t = self.query_time_embed(query_t).squeeze(1)
        q_c = self.query_cond_embed(query_c.squeeze(1))
        return self.query_proj(torch.cat([q_t, q_c], dim=-1))

    def forward(self, x, t, c, sex, lab_code, query_t, query_c, pad_mask=None, causal=True):
        # Use causal=True for autoregressive training
        Z = self.encode_sequence(x, t, sex, lab_code, pad_mask, causal=causal)
        query = self.process_query(query_t, query_c)
        combined = Z + query
        output = self.output_head(combined)
        return output[:, 0], output[:, 1]  # mu, log_var

class DualDecoder(TransformerEncoder):
    """Generates dual Gaussian distribution parameters (mu and log_var) for healthy and unhealthy conditions"""
    def __init__(self, d_model=128, nhead=4, num_layers=4, num_lab_codes=2):
        super().__init__(d_model, nhead, num_layers, num_lab_codes)

        self.query_time_embed = Time2Vec(d_model)
        self.query_proj = nn.Linear(d_model, d_model)

        def decoder_head():
            return nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, 2) 
            )

        self.healthy_head = decoder_head()
        self.unhealthy_head = decoder_head()
        
    def decode_healthy(self, Z, query_t):
        q_emb = self.query_time_embed(query_t).squeeze(1)  # [B, d_model]
        combined = Z + self.query_proj(q_emb)              # [B, d_model]
        return self.healthy_head(combined)                 # [B, 2]

    def decode_unhealthy(self, Z, query_t):
        q_emb = self.query_time_embed(query_t).squeeze(1)
        combined = Z + self.query_proj(q_emb)
        return self.unhealthy_head(combined)

    def decode_distributions(self, Z, query_t):
        q_emb = self.query_time_embed(query_t).squeeze(1)
        combined = Z + self.query_proj(q_emb)
        healthy_params = self.healthy_head(combined)
        unhealthy_params = self.unhealthy_head(combined)
        return healthy_params, unhealthy_params  # Each: [B, 2]

    def forward(self, x, t, sex, lab_code, query_t, pad_mask=None):
        Z_seq = self.encode_sequence(x, t, sex, lab_code, pad_mask)  # [B, L, d_model]
        Z = Z_seq.mean(dim=1)  # Simple mean pooling, shape: [B, d_model]
        return self.decode_distributions(Z, query_t)

class TimeConditionedTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, num_lab_codes=2, **kwargs):
        super().__init__()
        self.value_embed = nn.Linear(1, d_model)
        self.cond_embed = nn.Embedding(2, d_model)
        self.sex_embed = nn.Embedding(2, d_model)
        self.lab_code_embed = nn.Embedding(num_lab_codes, d_model)
        
        self.time_embed = Time2Vec(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output head for distribution parameters (mu, log_var)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 2)  # mu and log_var
        )

    def forward(self, x, t, c, sex, lab_code, query_t, query_c, pad_mask):
        # x: (B, T, 1)
        # t: (B, T, 1) 
        # c: (B, T) - raw indices
        # sex: (B, 1) - raw indices
        # lab_code: (B, 1) - raw indices
        # query_t: (B, 1, 1)
        # query_c: (B, 1) - raw indices
        
        B, T = c.shape
        
        # Embed static features
        sex_emb = self.sex_embed(sex).squeeze(1)  # (B, d_model)
        lab_code_emb = self.lab_code_embed(lab_code).squeeze(1)  # (B, d_model)
        s_emb = sex_emb + lab_code_emb  # (B, d_model)
        s_emb = s_emb.unsqueeze(1).expand(B, T, -1)  # (B, T, d_model)

        x_emb = self.value_embed(x)
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(c)  # (B, T, d_model)
        encoder_input = x_emb + t_emb + c_emb + s_emb

        # Encode sequence (keep batch_first=True)
        encoded = self.encoder(encoder_input, src_key_padding_mask=pad_mask)
        
        # Global average pooling over time dimension
        if pad_mask is not None:
            # Mask out padded positions
            mask = (~pad_mask).float().unsqueeze(-1)  # (B, T, 1)
            encoded = encoded * mask
            pooled = encoded.sum(dim=1) / mask.sum(dim=1)  # (B, d_model)
        else:
            pooled = encoded.mean(dim=1)  # (B, d_model)
        
        # Output distribution parameters
        output = self.output_head(pooled)  # (B, 2)
        mu = output[:, 0]      # (B,)
        log_var = output[:, 1] # (B,)
        return mu, log_var