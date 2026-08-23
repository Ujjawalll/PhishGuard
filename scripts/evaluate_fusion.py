import pandas as pd
import numpy as np
import os
import joblib
import json
import uuid
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from ml.rules.engine import RuleEngine
from ml.fusion.strategies import WeightedSumFusion

def evaluate_binary(y_true, y_pred, y_prob):
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
    model_version = config["model"]["production_model"]
    t_high = config["risk_thresholds"]["suspicious_to_high"]
    t_low = config["risk_thresholds"]["low_to_suspicious"]
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
    
    # Binary evaluation considers >= T_high as positive (phishing)
    fused_preds = (fused_probs >= t_high).astype(int)
    
    experiment_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    
    results = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "dataset": "random/val_features.csv",
        "split": "val",
        "feature_set": "lexical",
        "model": "xgboost",
        "model_version": model_version,
        "artifact_path": xgb_path,
        "fusion_strategy": "weighted_sum",
        "ML_weight": alpha,
        "rule_weight": 1.0 - alpha,
        "T_LOW": t_low,
        "T_HIGH": t_high,
        "random_seed": 42,
        "metrics_binary_high_risk": evaluate_binary(y_val, fused_preds, fused_probs)
    }
    
    print("\n=== Fusion Evaluation Report ===")
    print(json.dumps(results, indent=2))
    
    os.makedirs("experiments/reports", exist_ok=True)
    out_path = f"experiments/reports/fusion_evaluation_{experiment_id}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
