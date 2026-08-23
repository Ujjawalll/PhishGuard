import pandas as pd
import numpy as np
import joblib
import json
import os
from ml.rules.engine import RuleEngine
from ml.fusion.strategies import WeightedSumFusion
from sklearn.metrics import precision_score, recall_score, f1_score

def main():
    print("Loading production config...")
    with open("configs/production.json") as f:
        config = json.load(f)
        
    xgb_path = config["model"]["artifact_path"]
    alpha = config["fusion"]["ml_weight"]
    
    with open(os.path.join(xgb_path, "metadata.json")) as f:
        feature_cols = json.load(f)["features"]
        
    val_df = pd.read_csv("ml/data/features/random/val_features.csv")
    y_val = val_df['label'].values
    
    print("Extracting rule and ML scores...")
    rule_engine = RuleEngine()
    
    pipeline = joblib.load(os.path.join(xgb_path, "model.joblib"))
    ml_probs = pipeline.predict_proba(val_df[feature_cols])[:, 1]
    
    rule_scores = []
    features_dicts = val_df[feature_cols].to_dict('records')
    for url, features in zip(val_df['url'], features_dicts):
        res = rule_engine.evaluate(url, features)
        rule_scores.append(res["normalized_score"])
    rule_scores = np.array(rule_scores)
    
    fusion = WeightedSumFusion(alpha=alpha, threshold=0.5)
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
            "threshold": round(t, 2),
            "precision": precision_score(y_val, preds, zero_division=0),
            "recall": recall_score(y_val, preds, zero_division=0),
            "f1": f1_score(y_val, preds, zero_division=0),
            "fpr": fpr
        })
        
    res_df = pd.DataFrame(results)
    print("\nTop Thresholds by F1 (Candidate for T_high):")
    print(res_df.sort_values('f1', ascending=False).head(10))
    
    print("\nThresholds ensuring FPR < 0.01 (Candidate for T_low):")
    print(res_df[res_df['fpr'] < 0.01].sort_values('recall', ascending=False).head(10))
    
    os.makedirs("experiments/reports", exist_ok=True)
    res_df.to_csv("experiments/reports/threshold_sweep.csv", index=False)
    print("\nSaved to experiments/reports/threshold_sweep.csv")

if __name__ == "__main__":
    main()
