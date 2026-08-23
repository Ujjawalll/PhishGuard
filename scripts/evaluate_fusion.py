import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from ml.rules.engine import RuleEngine
from ml.fusion.strategies import WeightedSumFusion

def evaluate(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = cm[0][0], 0, 0, 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "fpr": fpr,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }

def main():
    print("Loading production config...")
    with open("configs/production.json") as f:
        config = json.load(f)
        
    xgb_path = config["model"]["artifact_path"]
    t_high = config["risk_thresholds"]["suspicious_to_high"]
    alpha = config["fusion"]["ml_weight"]
    
    print("Loading Validation Data...")
    val_df = pd.read_csv("ml/data/features/random/val_features.csv")
    
    with open(os.path.join(xgb_path, "metadata.json")) as f:
        feature_cols = json.load(f)["features"]
        
    print(f"Loaded ML model from: {xgb_path}")
    ml_model = joblib.load(os.path.join(xgb_path, "model.joblib"))
    rule_engine = RuleEngine()
    
    print("Generating base predictions...")
    X = val_df[feature_cols]
    ml_prob = ml_model.predict_proba(X)[:, 1]
    
    rule_scores = []
    features_dicts = X.to_dict('records')
    for url, features in zip(val_df['url'], features_dicts):
        res = rule_engine.evaluate(url, features)
        rule_scores.append(res['normalized_score'])
        
    rule_scores = np.array(rule_scores)
    y_val = val_df['label'].values
    
    fusion = WeightedSumFusion(alpha=alpha, threshold=t_high)
    fused_probs = fusion.predict_proba(ml_prob, rule_scores)
    fused_preds = (fused_probs >= t_high).astype(int)
    
    results = {
        "dataset": "random/val_features.csv",
        "model": xgb_path,
        "fusion_strategy": "weighted_sum",
        "ml_weight": alpha,
        "rule_weight": 1.0 - alpha,
        "T_high": t_high,
        "metrics": evaluate(y_val, fused_preds, fused_probs)
    }
    
    print("\n=== Fusion Evaluation Report ===")
    print(json.dumps(results, indent=2))
    
    os.makedirs("experiments/reports", exist_ok=True)
    with open("experiments/reports/fusion_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to experiments/reports/fusion_evaluation.json")

if __name__ == "__main__":
    main()
