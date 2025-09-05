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

def process_patient_sequences_fast(df, REFERENCE_INTERVALS, verbose=False):
    df = df.copy()

    # 1) Clean + types (categorical speeds groupby & vocab)
    df['test_name'] = df['test_name'].fillna('NA').astype('category')
    df['time'] = pd.to_datetime(df['time'], utc=False, errors='coerce')
    df['numeric_value'] = pd.to_numeric(df['numeric_value'], errors='coerce').astype(np.float32)
    df['gender_concept_id'] = df['gender_concept_id'].astype(np.int32)

    # 2) Prebuild test vocab (name -> id) and lab_code once
    cats = df['test_name'].cat.categories
    TEST_VOCAB = {name: i for i, name in enumerate(cats)}
    df['lab_code'] = df['test_name'].cat.codes.astype(np.int32)

    # 3) Build reference mu/var table once and merge (avoids per-group dict lookups)
    #    Expect REFERENCE_INTERVALS[test_name][sex_key] = (low, high, anything)
    rows = []
    for test, sexmap in REFERENCE_INTERVALS.items():
        for sex_key, (low, high, _) in sexmap.items():
            mu = (low + high) / 2.0
            var = ((high - low) / 4.0) ** 2
            rows.append((test, sex_key, mu, var))
    ref_df = pd.DataFrame(rows, columns=['test_name', 'sex_key', 'ref_mu', 'ref_var'])
    ref_df['test_name'] = ref_df['test_name'].astype('category')
    # Map gender_concept_id -> sex_key (adjust mapping to your data)
    df['sex_key'] = np.where(df['gender_concept_id'].to_numpy() == 1, 'F', 'M').astype('category')
    df = df.merge(ref_df, on=['test_name', 'sex_key'], how='left')

    # 4) Vectorized "condition" (replace df.apply(is_normal, axis=1))
    #    If your is_normal uses a different rule, adjust here.
    #    Inside-range -> 1, else 0:
    #    (If you want the opposite or a multi-class flag, change accordingly.)
    # NOTE: if a test/sex has no ref row, ref_mu/var will be NaN; treat as abnormal (0).
    low = df[['test_name','sex_key']].merge(
        pd.DataFrame(
            [(t, s, REFERENCE_INTERVALS[t][s][0]) for t in REFERENCE_INTERVALS for s in REFERENCE_INTERVALS[t]],
            columns=['test_name','sex_key','low']
        ).astype({'test_name':'category'}), on=['test_name','sex_key'], how='left'
    )['low'].to_numpy()
    high = df[['test_name','sex_key']].merge(
        pd.DataFrame(
            [(t, s, REFERENCE_INTERVALS[t][s][1]) for t in REFERENCE_INTERVALS for s in REFERENCE_INTERVALS[t]],
            columns=['test_name','sex_key','high']
        ).astype({'test_name':'category'}), on=['test_name','sex_key'], how='left'
    )['high'].to_numpy()
    val = df['numeric_value'].to_numpy()
    cond = (val >= np.nan_to_num(low, nan=np.inf*-1)) & (val <= np.nan_to_num(high, nan=np.inf))
    df['condition'] = cond.astype(np.int32)

    # 5) Sort once so groups are contiguous; observed=True avoids unused category overhead
    df.sort_values(['subject_id', 'test_name', 'time'], inplace=True, kind='mergesort')
    grp = df.groupby(['subject_id', 'test_name'], sort=False, observed=True)

    # 6) Keep only groups with >=3 rows (vectorized)
    keep_mask = grp['time'].transform('size') >= 3
    df = df[keep_mask]

    # 7) Relative time in days from each group's first timestamp (vectorized)
    first_time = grp['time'].transform('first')
    df['t_days'] = ((df['time'] - first_time).dt.total_seconds() / 86400.0).astype(np.float32)

    # 8) Build sequences with minimal Python per group (numpy views, no per-row ops)
    sequences = []
    for (pid, _), g in df.groupby(['subject_id', 'test_name'], sort=False, observed=True):
        # contiguous slices; avoid copies where possible
        t = g['t_days'].to_numpy(dtype=np.float32, copy=False)
        x = g['numeric_value'].to_numpy(dtype=np.float32, copy=False)
        c = g['condition'].to_numpy(dtype=np.int32, copy=False)

        if t.size < 3:
            continue

        sex = int(g['gender_concept_id'].iat[0])
        lab = int(g['lab_code'].iat[0])
        ref_mu = np.float32(g['ref_mu'].iat[0]) if 'ref_mu' in g else np.float32(np.nan)
        ref_var = np.float32(g['ref_var'].iat[0]) if 'ref_var' in g else np.float32(np.nan)

        sequences.append({
            "x": x[:-1],
            "t": t[:-1],
            "c": c[:-1],
            "s": np.array([sex, lab], dtype=np.int32),
            "query_t": t[-1:].astype(np.float32),
            "query_c": c[-1:].astype(np.int32),
            "target": x[-1:].astype(np.float32),
            "ref_mu": np.array([ref_mu], dtype=np.float32),
            "ref_var": np.array([ref_var], dtype=np.float32),
            "subject_id": pid,
        })

    if verbose and sequences:
        print(f"Kept {len(sequences)} sequences (groups with ≥3 rows).")
        sample = sequences[0]
        print(f"  Values: {sample['x'][:5]}  Times: {sample['t'][:5]}  Cond: {sample['c'][:5]}")
        print(f"  Target: {sample['target'][0]}  Query_c: {sample['query_c'][0]}  Query_t: {sample['query_t'][0]}")
        print(f"  Ref mu/var: {sample['ref_mu'][0]}, {sample['ref_var'][0]}")

    # Optionally keep the global for backward-compat
    TEST_VOCAB = {name: i for i, name in enumerate(df['test_name'].cat.categories.unique())}

    return sequences, TEST_VOCAB

class TimeSeriesForecastingDataset(Dataset):
    def __init__(self, sequences):
        self.data = sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Convert numpy arrays to tensors
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

    # Pad sequences and stack tensors
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

    # Create padding mask
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

    # Create DataLoaders with GPU support
    train_loader = DataLoader(
        TimeSeriesForecastingDataset(train_seq),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        TimeSeriesForecastingDataset(val_seq), 
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        TimeSeriesForecastingDataset(test_seq), 
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

def main():
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = pd.read_csv("../../data/processed/lab_measurements.csv")
    train_loader, val_loader, test_loader = create_dataloaders(df, batch_size=16)

    # Test data loading to GPU
    for x, t, c, s, query_t, query_c, query_s, target, pad_mask in train_loader:
        # Move tensors to GPU if available
        x = x.to(device)
        t = t.to(device) 
        c = c.to(device)
        s = s.to(device)
        query_t = query_t.to(device)
        query_c = query_c.to(device)
        query_s = query_s.to(device)
        target = target.to(device)
        pad_mask = pad_mask.to(device)

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
        print(f"Device: {x.device}")
        break

if __name__ == "__main__":
    main()