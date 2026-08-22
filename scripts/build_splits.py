import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit
import json

DATA_DIR = "ml/data/processed"
SPLITS_DIR = "ml/data/splits"

def split_stratified_group(df, test_size, random_state=42):
    """Helper to do a GroupShuffleSplit but stratified by label manually by splitting each class separately."""
    df_phish = df[df['label'] == 1].copy()
    df_legit = df[df['label'] == 0].copy()
    
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    train_p_idx, test_p_idx = next(gss.split(df_phish, groups=df_phish['domain']))
    train_l_idx, test_l_idx = next(gss.split(df_legit, groups=df_legit['domain']))
    
    train_df = pd.concat([df_phish.iloc[train_p_idx], df_legit.iloc[train_l_idx]])
    test_df = pd.concat([df_phish.iloc[test_p_idx], df_legit.iloc[test_l_idx]])
    
    # Shuffle the resulting dataframes
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return train_df, test_df

def create_random_split(df):
    """Domain-aware random split to prevent domain leakage, stratified by label"""
    
    train_df, test_df = split_stratified_group(df, test_size=0.2, random_state=42)
    
    # 0.125 of 0.8 is 0.1 of total
    train_df, val_df = split_stratified_group(train_df, test_size=0.125, random_state=42)
    
    split_dir = os.path.join(SPLITS_DIR, "random")
    os.makedirs(split_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(split_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(split_dir, "test.csv"), index=False)
    
    return {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "train_phish": int(train_df['label'].sum()),
        "val_phish": int(val_df['label'].sum()),
        "test_phish": int(test_df['label'].sum())
    }

def create_cross_dataset_split(df):
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
    df = pd.read_csv(path)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    
    manifest = {}
    manifest['random'] = create_random_split(df)
    manifest['cross_dataset'] = create_cross_dataset_split(df)
    manifest['temporal'] = "Not applicable due to lack of per-row timestamps."
    
    with open(os.path.join(SPLITS_DIR, "splits_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    main()
