import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit
import json

DATA_DIR = "ml/data/processed"
SPLITS_DIR = "ml/data/splits"

def create_random_split(df):
    """Domain-aware random split to prevent domain leakage"""
    # Use GroupShuffleSplit to ensure domains don't cross boundaries
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['domain']))
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    # Further split train into train/val
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42) # 0.125 of 0.8 is 0.1 of total
    train_idx2, val_idx = next(gss_val.split(train_df, groups=train_df['domain']))
    
    val_df = train_df.iloc[val_idx].copy()
    train_df = train_df.iloc[train_idx2].copy()
    
    split_dir = os.path.join(SPLITS_DIR, "random")
    os.makedirs(split_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(split_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(split_dir, "test.csv"), index=False)
    
    return {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df)
    }

def create_cross_dataset_split(df):
    """
    Simulate cross-dataset split. 
    Here, train on most of OpenPhish+Tranco, but we ideally need another source.
    For this mockup, we'll divide based on a hash of the URL to simulate different distributions.
    In a real scenario, this would be train on PhishTank, test on OpenPhish.
    """
    # Since we only have OpenPhish and Tranco, we'll create a synthetic cross-dataset split
    # by holding out domains starting with specific letters for testing.
    split_dir = os.path.join(SPLITS_DIR, "cross_dataset")
    os.makedirs(split_dir, exist_ok=True)
    
    test_mask = df['domain'].str.match(r'^[a-d0-9]', na=False)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()
    
    train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(split_dir, "test.csv"), index=False)
    
    return {
        "train": len(train_df),
        "test": len(test_df)
    }

def main():
    path = os.path.join(DATA_DIR, "cleaned_dataset.csv")
    if not os.path.exists(path):
        print(f"Cleaned dataset not found at {path}")
        return
        
    df = pd.read_csv(path)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    
    manifest = {}
    
    print("Creating Random Split (domain-aware)...")
    manifest['random'] = create_random_split(df)
    
    print("Creating Cross-Dataset Split...")
    manifest['cross_dataset'] = create_cross_dataset_split(df)
    
    # For temporal split, we normally need timestamps.
    # Since our datasets are snapshot downloads without per-row timestamps, 
    # we'll skip the temporal split or synthesize one if needed later.
    manifest['temporal'] = "Not applicable due to lack of per-row timestamps in current snapshot."
    
    with open(os.path.join(SPLITS_DIR, "splits_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("Splits created successfully.")

if __name__ == "__main__":
    main()
