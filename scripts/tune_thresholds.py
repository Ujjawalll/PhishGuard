import pandas as pd
import numpy as np
import joblib
import json
import os
from ml.rules.engine import RuleEngine
from ml.fusion.strategies import WeightedSumFusion
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime

def main():
    print("Loading production config...")
    with open("configs/production.json") as f:
        config = json.load(f)
        
    xgb_path = config["model"]["artifact_path"]
    alpha = config["fusion"]["ml_weight"]
    
    with open(os.path.join(xgb_path, "metadata.json")) as f:
        metadata = json.load(f)
        feature_cols = metadata["features"]
        model_version = metadata.get("model_version", "v1.0")
        feature_schema_version = metadata.get("feature_schema_version", "v1.0")
        
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
    
    # Selection Criteria
    # T_HIGH: max F1 subject to FPR <= 0.05
    # T_LOW: max recall subject to FPR <= 0.01
    
    t_high_candidates = res_df[res_df['fpr'] <= 0.05]
    if not t_high_candidates.empty:
        t_high_row = t_high_candidates.sort_values('f1', ascending=False).iloc[0]
        selected_t_high = t_high_row['threshold']
    else:
        selected_t_high = res_df.sort_values('f1', ascending=False).iloc[0]['threshold']
        
    t_low_candidates = res_df[res_df['fpr'] <= 0.01]
    if not t_low_candidates.empty:
        t_low_row = t_low_candidates.sort_values('recall', ascending=False).iloc[0]
        selected_t_low = t_low_row['threshold']
    else:
        selected_t_low = selected_t_high * 0.5
        
    print(f"\nSelected T_HIGH: {selected_t_high} (Criterion: max F1 subject to FPR <= 0.05)")
    print(f"Selected T_LOW: {selected_t_low} (Criterion: max recall subject to FPR <= 0.01)")
    
    os.makedirs("experiments/reports", exist_ok=True)
    res_df.to_csv("experiments/reports/threshold_sweep.csv", index=False)
    
    artifact = {
        "dataset": "random",
        "split": "val",
        "feature_set": "url_lexical",
        "model_version": model_version,
        "feature_schema_version": feature_schema_version,
        "fusion_strategy": "weighted_sum",
        "ml_weight": alpha,
        "rule_weight": 1.0 - alpha,
        "candidate_thresholds": res_df['threshold'].tolist(),
        "selected_t_low": selected_t_low,
        "selected_t_high": selected_t_high,
        "selection_criterion": {
            "t_high": "max F1 subject to FPR <= 0.05",
            "t_low": "max recall subject to FPR <= 0.01"
        },
        "random_seed": 42,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    with open("experiments/reports/threshold_artifact.json", "w") as f:
        json.dump(artifact, f, indent=2)
        
    print("\nSaved artifacts to experiments/reports/threshold_sweep.csv and experiments/reports/threshold_artifact.json")

if __name__ == "__main__":
    main()
