import torch
import torch.nn as nn

class SinusoidalPositionalEmbedding(nn.Embedding):
    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out):
        n_pos, dim = out.shape
        position_enc = torch.tensor([
            [pos / (10000 ** (2 * (j // 2) / dim)) for j in range(dim)] for pos in range(n_pos)
        ])
        out.detach_()
        out.requires_grad = False
        out[:, :dim // 2] = torch.sin(position_enc[:, 0::2])
        out[:, dim // 2:] = torch.cos(position_enc[:, 1::2])
        return out

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(seq_len, device=self.weight.device)
        return super().forward(positions)

class DateTimeEmbedding(nn.Module):
    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__()
        self.embed_year = SinusoidalPositionalEmbedding(num_positions, embedding_dim, padding_idx)
        self.embed_month = nn.Embedding(13, embedding_dim)
        self.embed_day = nn.Embedding(32, embedding_dim)
        self.embed_time = nn.Embedding(25, embedding_dim)

    def forward(self, input):
        year = self.embed_year(input["year"])
        month = self.embed_month(input["month"])
        day = self.embed_day(input["day"])
        hour = self.embed_time(input["time"])
        return year + month + day + hour

class TransformerEncoder(nn.Transformer):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, nout, dropout=0.5, position="time"):
        super(TransformerEncoder, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, 
                                               num_encoder_layers=nlayers, num_decoder_layers=1, batch_first=True)

        self.src_mask = None
        self.ntoken = ntoken
        self.ninp = ninp
        self.input_proj = nn.Linear(1, ninp)
        self.position = position
        self.pos_encoder = DateTimeEmbedding(self.ntoken, ninp)

        self.layer_norm = nn.LayerNorm(ninp)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(ninp, nout)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, src_key_padding_mask, dates=None, target_time=None, has_mask=True, device=None):
        src = self.input_proj(src)
        
        if has_mask:
            if self.src_mask is None or self.src_mask.size(0) != src.shape[1]:
                mask = self._generate_square_subsequent_mask(src.shape[1]).to(device)
                self.src_mask = mask
            else:
                self.src_mask = self.src_mask.to(device)
        else:
            self.src_mask = None

        if self.position == "time": 
            time_emb = self.pos_encoder(dates)
        else: 
            time_emb = self.pos_encoder(src)

        src = self.dropout(src + time_emb)
        src = self.layer_norm(src)
        
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.float()
        
        output = self.encoder(src, src_key_padding_mask=src_key_padding_mask, mask=self.src_mask)
        output = output[:, -1, :]
        return self.decoder(output)
