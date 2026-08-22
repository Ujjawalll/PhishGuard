import pandas as pd
import os
from tqdm import tqdm
from ml.features.lexical import extract_lexical_features

SPLITS_DIR = "ml/data/splits/random"
FEAT_DIR = "ml/data/features/random"

def process_split(split_name):
    df = pd.read_csv(os.path.join(SPLITS_DIR, f"{split_name}.csv"))
    print(f"Extracting features for {split_name} ({len(df)} rows)...")
    
    features_list = []
    for url in tqdm(df['url']):
        features = extract_lexical_features(url)
        features_list.append(features)
        
    feat_df = pd.DataFrame(features_list)
    
    # Merge with original label and domain
    feat_df['label'] = df['label']
    feat_df['domain'] = df['domain']
    feat_df['url'] = df['url']
    
    os.makedirs(FEAT_DIR, exist_ok=True)
    feat_df.to_csv(os.path.join(FEAT_DIR, f"{split_name}_features.csv"), index=False)

def main():
    for split in ["train", "val", "test"]:
        process_split(split)
        
if __name__ == "__main__":
    main()
