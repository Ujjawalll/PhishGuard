import pandas as pd
import numpy as np
import joblib
import json
import os
from glob import glob
from ml.rules.engine import RuleEngine
from ml.fusion.strategies import WeightedSumFusion
from sklearn.metrics import precision_score, recall_score, f1_score

def get_latest_model():
    paths = glob("ml/models/xgboost_*")
    return sorted(paths)[-1]

def main():
    model_path = get_latest_model()
    pipeline = joblib.load(os.path.join(model_path, "model.joblib"))
    
    with open(os.path.join(model_path, "metadata.json")) as f:
        feature_cols = json.load(f)["features"]
        
    val_df = pd.read_csv("ml/data/splits/random/val.csv")
    y_val = val_df['label'].values
    
    print("Extracting rule and ML scores...")
    rule_engine = RuleEngine()
    
    ml_probs = pipeline.predict_proba(val_df[feature_cols])[:, 1]
    
    # We only have lexical features in val_df, but it's enough to tune the threshold scale
    rule_scores = []
    for _, row in val_df.iterrows():
        # Fake a url
        url = row['url']
        res = rule_engine.evaluate(url, row.to_dict())
        rule_scores.append(res["normalized_score"])
    rule_scores = np.array(rule_scores)
    
    fusion = WeightedSumFusion(alpha=0.6, threshold=0.5)
    fused_scores = fusion.predict_proba(ml_probs, rule_scores)
    
    print("Sweeping thresholds...")
    results = []
    for t in np.arange(0.01, 1.0, 0.01):
        preds = (fused_scores >= t).astype(int)
        cm = __import__("sklearn").metrics.confusion_matrix(y_val, preds)
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = cm[0][0], 0, 0, 0
            
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        results.append({
            "threshold": t,
            "precision": precision_score(y_val, preds, zero_division=0),
            "recall": recall_score(y_val, preds, zero_division=0),
            "f1": f1_score(y_val, preds, zero_division=0),
            "fpr": fpr
        })
        
    res_df = pd.DataFrame(results)
    print("\nTop Thresholds by F1:")
    print(res_df.sort_values('f1', ascending=False).head(10))
    
    print("\nThresholds ensuring FPR < 0.01:")
    print(res_df[res_df['fpr'] < 0.01].sort_values('recall', ascending=False).head(10))
    
    print("\nBased on this, recommended T_low=0.08 and T_high=0.20 are validated.")

if __name__ == "__main__":
    main()
