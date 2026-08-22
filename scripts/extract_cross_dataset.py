import pandas as pd
import os
from tqdm import tqdm
from ml.features.lexical import extract_lexical_features

SPLIT_PATH = "ml/data/splits/cross_dataset.csv"
FEAT_DIR = "ml/data/features"

def process_split():
    if not os.path.exists(SPLIT_PATH):
        print(f"Skipping cross-dataset, not found: {SPLIT_PATH}")
        return
        
    df = pd.read_csv(SPLIT_PATH)
    print(f"Extracting features for cross-dataset ({len(df)} rows)...")
    
    features_list = []
    for url in tqdm(df['url']):
        features_list.append(extract_lexical_features(url))
        
    feat_df = pd.DataFrame(features_list)
    feat_df['label'] = df['label']
    feat_df['domain'] = df['domain']
    feat_df['url'] = df['url']
    
    os.makedirs(FEAT_DIR, exist_ok=True)
    feat_df.to_csv(os.path.join(FEAT_DIR, "cross_dataset_features.csv"), index=False)

if __name__ == "__main__":
    process_split()
