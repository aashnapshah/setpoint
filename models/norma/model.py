# Input:
#   - Past: sequence of (xᵢ, tᵢ, cᵢ)
#   - Static variable: s
#   - Future target condition/time: (t′, cₜ′)

# 1. Encoder:
#   - Eₓ(xᵢ): Value embedding
#   - Eₜ(tᵢ): Time embedding
#   - E_c(cᵢ): Condition embeddingq
#   - E_s(s): Static embedding (broadcasted or added)

#   → Token embedding: Eᵢ = Eₓ + Eₜ + E_c + E_s

# 2. Transformer encoder on E₁...Eₙ

# 3. Decoder:
#   - Query token: Eₜ(t′) + E_c(cₜ′) + E_s(s)
#   - Cross-attend to encoder output

# 4. Output head:
#   - FFN → Regression → Predict xₜ′

import torch
import torch.nn as nn
from decoders import DecoderFactory

class Time2Vec(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(1, 1)                      # Linear trend
        self.periodic = nn.Linear(1, d_model - 1)          # Periodic components

    def forward(self, t):  # t: (B, T, 1) — scalar times or deltas
        v_linear = self.linear(t)                          # (B, T, 1)
        v_periodic = torch.sin(self.periodic(t))           # (B, T, D-1)
        return torch.cat([v_linear, v_periodic], dim=-1)   # (B, T, D)

class TimeConditionedTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, num_lab_codes=2, decoder_type='nll', **decoder_kwargs):
        super().__init__()
        self.value_embed = nn.Linear(1, d_model)
        self.cond_embed = nn.Embedding(2, d_model)
        self.sex_embed = nn.Embedding(2, d_model)
        self.lab_code_embed = nn.Embedding(num_lab_codes, d_model)
        
        self.time_embed = Time2Vec(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Modular decoder head
        self.decoder_type = decoder_type
        self.output_head = DecoderFactory.create_decoder(decoder_type, d_model, **decoder_kwargs)

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

        encoder_input = encoder_input.permute(1, 0, 2)  # (T, B, D)
        memory = self.encoder(encoder_input, src_key_padding_mask=pad_mask)

        # Query embeddings
        q_t_emb = self.time_embed(query_t)
        q_c_emb = self.cond_embed(query_c)
        query = q_t_emb + q_c_emb  # (B, 1, D)
        query = query.permute(1, 0, 2)

        decoded = self.decoder(query, memory)
        
        # Handle different decoder output formats
        if self.decoder_type == 'mdn':
            # MDN expects (B, D) input
            out = decoded.permute(1, 0, 2).squeeze(1)  # (B, D)
            return self.output_head(out)  # Returns (pi, mu, log_var) tuple
        else:
            # Other decoders can handle the transformer output directly
            out = self.output_head(decoded)  # (1, B, D)
            out = out.permute(1, 0, 2)  # (B, 1, D)
            out = out.squeeze(1)  # (B, D)
            return out

