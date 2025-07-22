import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda": print(torch.cuda.get_device_name(0))

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

def extract_time_features(dt):
    return dt.year, dt.month, dt.day, dt.hour

def create_sequences(df):
    sequences = []
    max_len = 0
    
    for patient_id in df['subject_id'].unique():
        patient_data = df[df['subject_id'] == patient_id].copy()
        patient_data = patient_data.sort_values('time')
        
        if patient_data['time'].dtype == 'object':
            patient_data['time'] = pd.to_datetime(patient_data['time'])
        
        times = patient_data['time'].tolist()
        values = patient_data['numeric_value'].values

        if len(values) >= 2:
            years, months, days, hours = [], [], [], []
            for t in times:
                y, m, d, h = extract_time_features(t)
                years.append(y)
                months.append(m)
                days.append(d)
                hours.append(h)

            seq_len = len(values) - 1
            max_len = max(max_len, seq_len)

            sequences.append({
                'patient_id': patient_id,
                'values': values[:seq_len],
                'target': values[-1],
                'year': years[:seq_len],
                'month': months[:seq_len],
                'day': days[:seq_len],
                'hour': hours[:seq_len]
            })

    global MAX_SEQ_LEN
    MAX_SEQ_LEN = max_len
    return sequences

class HGBTimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.samples = sequences

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        x = torch.tensor(sample['values'], dtype=torch.float32).unsqueeze(-1)
        
        if len(x) < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - len(x)
            x = torch.cat([x, torch.zeros(pad_len, 1)], dim=0)
        
        time_dict = {
            "year": torch.tensor(sample['year'], dtype=torch.long),
            "month": torch.tensor(sample['month'], dtype=torch.long),
            "day": torch.tensor(sample['day'], dtype=torch.long),
            "time": torch.tensor(sample['hour'], dtype=torch.long)
        }
        
        if len(time_dict["year"]) < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - len(time_dict["year"])
            for key in time_dict:
                time_dict[key] = torch.cat([time_dict[key], torch.zeros(pad_len, dtype=torch.long)], dim=0)
        
        next_target = torch.tensor(sample['target'], dtype=torch.float32)
        return x, time_dict, next_target

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

    def forward(self, src, src_key_padding_mask, dates=None, has_mask=True, device=None):
        src = self.input_proj(src)
        
        if has_mask:
            if self.src_mask is None or self.src_mask.size(0) != src.shape[1]:
                mask = self._generate_square_subsequent_mask(src.shape[1]).to(device)
                self.src_mask = mask
            else:
                self.src_mask = self.src_mask.to(device)
        else:
            self.src_mask = None

        if self.position == "time": time_emb = self.pos_encoder(dates)
        else: time_emb = self.pos_encoder(src)

        src = self.dropout(src + time_emb)
        src = self.layer_norm(src)
        
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.float()
        
        output = self.encoder(src, src_key_padding_mask=src_key_padding_mask, mask=self.src_mask)
        output = output[:, -1, :]
        return self.decoder(output)

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

def main():
    data_path = '../../data/processed/lab_measurements.csv'
    measurements = pd.read_csv(data_path, parse_dates=['time']).query('test_name == "HGB"')
    sequences = create_sequences(measurements)
    train_data, val_data = train_test_split(sequences, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=42)

    train_dataset = HGBTimeSeriesDataset(train_data)
    val_dataset = HGBTimeSeriesDataset(val_data)
    test_dataset = HGBTimeSeriesDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"\nTesting dataloader...")
    for i, (x, time_dict, target) in enumerate(train_dataloader):
        print(f"Batch {i+1}")
        print(f"  x: {x.shape}, target: {target.shape}")
        print(f"  Time dict keys: {list(time_dict.keys())}")
        print(f"  Year shape: {time_dict['year'].shape}")
        print(f"  Patient ID: {sequences[i]['patient_id']}")  # Print patient ID
        print(f"  Sample values: {x[0, :5, 0].numpy()}")  # First 5 values of first sample
        print(f"  Sample years: {time_dict['year'][0, :5].numpy()}")  # First 5 years of first sample
        print(f"  Target value: {target[0].item()}")  # First target value
        if i == 0: break

    model = TransformerEncoder(
        ntoken=MAX_SEQ_LEN,
        ninp=64,
        nhead=4,
        nhid=128,
        nlayers=2,
        nout=1,
        dropout=0.1,
        position="time"
    )
    
    metrics = train(model, train_dataloader, val_dataloader, epochs=10, learning_rate=0.001, device=device)
    print(metrics)
    
    model.eval()
    with torch.no_grad():
        x, time_dict, target = next(iter(test_dataloader))
        x = x.to(device)
        target = target.to(device)
        for k in time_dict:
            time_dict[k] = time_dict[k].to(device)
        
        src_key_padding_mask = (x.squeeze(-1) == 0).to(device)
        pred = model(x, src_key_padding_mask, dates=time_dict, device=device)
        
        print(f"Model test results:")
        print(f"  Input shape: {x.shape}")
        print(f"  Prediction shape: {pred.shape}")
        print(f"  Sample predictions vs targets:")
        for i in range(min(5, len(pred))):
            print(f"    Sample {i}: Predicted {pred[i].item():.2f}, Actual {target[i].item():.2f}")

if __name__ == "__main__":
    main()