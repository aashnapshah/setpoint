import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import sys

sys.path.append('../../')
from process.config import REFERENCE_INTERVALS

def is_normal(row):
    sex = 'F' if row['gender_concept_id'] == 1 else 'M'
    test_name = row['test_name']
    ref_range = REFERENCE_INTERVALS[test_name][sex]
    return ref_range[0] < row['numeric_value'] < ref_range[1]

def process_patient_sequences(df):
    df['test_name'] = df['test_name'].fillna('NA')
    df['condition'] = df.apply(is_normal, axis=1)

    global TEST_VOCAB
    TEST_VOCAB = {test_name: i for i, test_name in enumerate(df['test_name'].unique())}
    print(f"Test vocabulary: {TEST_VOCAB}")
    
    sequences = []

    for (pid, test_name), group in df.groupby(["subject_id", "test_name"]):
        if len(group) < 3:
            continue

        group = group.sort_values("time")
        times = pd.to_datetime(group["time"])
        t = np.array([(t - times.iloc[0]).total_seconds() / (24 * 3600) for t in times], dtype=np.float32)
        x = group["numeric_value"].values.astype(np.float32)
        c = group["condition"].values.astype(np.int32)
        sex = group["gender_concept_id"].values[0]
        lab = TEST_VOCAB[test_name]  # Now we know test_name from groupby

        # reference stats - now correctly for this specific test
        sex_key = 'F' if sex == 1 else 'M'
        low, high, _ = REFERENCE_INTERVALS[test_name][sex_key]
        ref_mu = (low + high) / 2.0
        ref_var = ((high - low) / 4.0) ** 2

        # Align query time with the target timestamp
        future_time = t[-1]
        future_condition = c[-1]
        target = x[-1]

        sequences.append({
            "x": x[:-1],
            "t": t[:-1],
            "c": c[:-1],
            "s": [sex, lab],
            "query_t": np.array([future_time], dtype=np.float32),
            "query_c": np.array([future_condition], dtype=np.int32),
            "target": np.array([target], dtype=np.float32),
            "ref_mu": np.array([ref_mu], dtype=np.float32),
            "ref_var": np.array([ref_var], dtype=np.float32),
            "subject_id": pid,  # Add subject ID
        })

    print(f"Kept {len(sequences)} patients with 3+ data points")
    print(f"\nData processing summary:")
    print(f"Total patients: {len(sequences)}")
    print(f"Sample patient data:")
    if sequences:
        sample = sequences[0]
        print(f"  Values: {sample['x'][:5]}...")
        print(f"  Times: {sample['t'][:5]}...")
        print(f"  Conditions: {sample['c'][:5]}...")
        print(f"  Target: {sample['target'][0]}")
        print(f"  Query condition: {sample['query_c'][0]}")
        print(f"  Query time: {sample['query_t'][0]}")
        print(f"  Ref mu/var: {sample['ref_mu'][0]}, {sample['ref_var'][0]}")

    return sequences

class TimeSeriesForecastingDataset(Dataset):
    def __init__(self, sequences):
        self.data = sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = torch.from_numpy(item["x"]).float().unsqueeze(-1)
        t = torch.from_numpy(item["t"]).float().unsqueeze(-1)
        c = torch.from_numpy(item["c"]).long()

        sex = torch.tensor([item["s"][0]], dtype=torch.long)
        lab_code = torch.tensor([item["s"][1]], dtype=torch.long)

        query_t = torch.from_numpy(item["query_t"]).unsqueeze(0).unsqueeze(-1)
        query_c = torch.from_numpy(item["query_c"]).unsqueeze(0)
        target = torch.tensor(item["target"][0], dtype=torch.float32)

        ref_mu = torch.tensor(item["ref_mu"][0], dtype=torch.float32)
        ref_var = torch.tensor(item["ref_var"][0], dtype=torch.float32)

        subject_id = item["subject_id"]

        return x, t, c, sex, lab_code, query_t, query_c, target, ref_mu, ref_var, subject_id

def collate_fn(batch):
    x, t, c, sex, lab_code, q_t, q_c, y, ref_mu, ref_var, subject_ids = zip(*batch)

    x = pad_sequence(x, batch_first=True)
    t = pad_sequence(t, batch_first=True)
    c = pad_sequence(c, batch_first=True)

    sex = torch.stack(sex)
    lab_code = torch.stack(lab_code)
    q_t = torch.cat(q_t, dim=0)
    q_c = torch.cat(q_c, dim=0)
    y = torch.stack(y)
    ref_mu = torch.stack(ref_mu)
    ref_var = torch.stack(ref_var)

    lengths = [seq.shape[0] for seq in x]
    max_len = x.shape[1]
    pad_mask = torch.ones(len(lengths), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        pad_mask[i, :l] = False

    return x, t, c, sex, lab_code, q_t, q_c, y, ref_mu, ref_var, pad_mask, subject_ids 

def create_dataloaders(df, batch_size=16):
    sequences = process_patient_sequences(df)
    train_seq, test_seq = train_test_split(sequences, test_size=0.2, random_state=42)
    train_seq, val_seq = train_test_split(train_seq, test_size=0.1, random_state=42)

    train_loader = DataLoader(TimeSeriesForecastingDataset(train_seq),
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(TimeSeriesForecastingDataset(val_seq), batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(TimeSeriesForecastingDataset(test_seq), batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader

def main():
    df = pd.read_csv("../../data/processed/lab_measurements.csv")
    train_loader, val_loader, test_loader = create_dataloaders(df, batch_size=16)

    for x, t, c, s, query_t, query_c, query_s, target, pad_mask in train_loader:
        print("Shapes:")
        print("x:", x.shape)
        print("t:", t.shape)
        print("c:", c.shape)
        print("s:", s.shape)
        print("query_t:", query_t.shape)
        print("query_c:", query_c.shape)
        print("query_s:", query_s.shape)
        print("target:", target.shape)
        print("pad_mask:", pad_mask.shape)
        break

if __name__ == "__main__":
    main()