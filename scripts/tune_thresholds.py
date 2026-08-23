import pandas as pd
import numpy as np
import joblib
import json
from glob import glob
from sklearn.metrics import f1_score, precision_score, recall_score
from ml.features.schema import FEATURE_SCHEMA

def main():
    val_df = pd.read_csv("ml/data/features/random/val_features.csv")
    X_val = val_df[FEATURE_SCHEMA]
    y_val = val_df['label']
    
    paths = glob("ml/models/xgboost_*")
    latest_path = sorted(paths)[-1]
    pipeline = joblib.load(latest_path + "/model.joblib")
    
    ml_probs = pipeline.predict_proba(X_val)[:, 1]
    
    # We will simulate rule score just as ml_prob for now, or just use ml_prob to tune threshold for ML
    best_f1 = 0
    best_t = 0
    for t in np.arange(0.01, 0.99, 0.01):
        preds = (ml_probs >= t).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    preds = (ml_probs >= best_t).astype(int)
    p = precision_score(y_val, preds)
    r = recall_score(y_val, preds)
    print(f"Best Threshold for ML only: {best_t:.2f}")
    print(f"F1: {best_f1:.4f} | Precision: {p:.4f} | Recall: {r:.4f}")

if __name__ == "__main__":
    main()
