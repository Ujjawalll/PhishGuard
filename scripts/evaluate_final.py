import pandas as pd
import numpy as np
import os
import time
import joblib
import json
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix

from ml.features.lexical import extract_lexical_features
from ml.rules.engine import RuleEngine
from ml.fusion.strategies import WeightedSumFusion

def get_latest_model():
    with open("configs/production.json") as f:
        config = json.load(f)
    return config["model"]["artifact_path"]

def evaluate(y_true, y_pred, y_prob, latencies=None):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = cm[0][0], 0, 0, 0 # Fallback
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    res = {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "FPR": float(fpr),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob))
    }
    if latencies:
        res["Latency"] = float(np.mean(latencies) * 1000)
    return res

def main():
    print("Loading test data...")
    test_df = pd.read_csv("ml/data/splits/cross_dataset/test.csv")
    y_test = test_df['label'].values
    
    print("Loading model and rules...")
    model_path = get_latest_model()
    pipeline = joblib.load(os.path.join(model_path, "model.joblib"))
    
    with open(os.path.join(model_path, "metadata.json")) as f:
        metadata = json.load(f)
        feature_cols = metadata["features"]
        model_version = metadata.get("model_version", "v1.0")
        feature_schema_version = metadata.get("feature_schema_version", "v1.0")
        
    rule_engine = RuleEngine()
    
    with open("configs/production.json") as f:
        config = json.load(f)
    alpha = config["fusion"]["ml_weight"]
    t_high = config["risk_thresholds"]["suspicious_to_high"]
    t_low = config["risk_thresholds"]["low_to_suspicious"]
    
    # Load ML-only and Rule-only thresholds
    ml_thresh = 0.22 # Fallback
    rule_thresh = 0.5 # Fallback
    
    if os.path.exists("experiments/reports/threshold_artifact.json"):
        with open("experiments/reports/threshold_artifact.json") as f:
            thresh_artifact = json.load(f)
            ml_thresh = thresh_artifact.get("ml_only_threshold", ml_thresh)
            
    if os.path.exists("experiments/results/rule_baseline.txt"):
        with open("experiments/results/rule_baseline.txt") as f:
            for line in f:
                if line.startswith("T_RULE:"):
                    rule_thresh = float(line.strip().split(": ")[1])
                    break
    
    fusion = WeightedSumFusion(alpha=alpha, threshold=t_high)
    
    print("Generating predictions...")
    
    ml_preds, ml_probs, ml_lats = [], [], []
    rule_preds, rule_probs, rule_lats = [], [], []
    fuse_preds, fuse_probs, fuse_lats = [], [], []
    
    fp_diagnostics = []
    
    for url, label in zip(test_df['url'], y_test):
        t0 = time.perf_counter()
        features = extract_lexical_features(url)
        t_feat = time.perf_counter()
        
        # Rule prediction
        res = rule_engine.evaluate(url, features)
        r_score = res['normalized_score']
        t_rule = time.perf_counter()
        
        # ML prediction
        df_single = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}])
        m_prob = pipeline.predict_proba(df_single)[0][1]
        t_ml = time.perf_counter()
        
        # Fusion prediction
        f_prob = fusion.predict_proba(np.array([m_prob]), np.array([r_score]))[0]
        t_fuse = time.perf_counter()
        
        ml_probs.append(m_prob)
        ml_preds.append(int(m_prob >= ml_thresh))
        ml_lats.append(t_ml - t_rule + (t_feat - t0))
        
        rule_probs.append(r_score)
        rule_preds.append(int(r_score >= rule_thresh))
        rule_lats.append(t_rule - t_feat + (t_feat - t0))
        
        fuse_probs.append(f_prob)
        fuse_preds.append(int(f_prob >= t_high))
        fuse_lats.append(t_fuse - t0)
        
        # False Positive Diagnostics
        if label == 0 and f_prob >= t_high:
            fp_diagnostics.append({
                "url": url,
                "label": int(label),
                "rule_score": r_score,
                "category_scores": res.get("category_scores", {}),
                "triggered_rules": [r["rule_id"] for r in res.get("triggered_rules", [])],
                "ml_probability": m_prob,
                "fused_score": f_prob,
                "deep_features": {k:v for k,v in features.items() if k not in feature_cols},
                "thresholds": {"T_LOW": t_low, "T_HIGH": t_high, "ML_ONLY": ml_thresh, "RULE_ONLY": rule_thresh},
                "model_version": model_version
            })
            
    print("\nCalculating metrics...")
    metrics = {
        "ML_Only": evaluate(y_test, ml_preds, ml_probs, ml_lats),
        "Rules_Only": evaluate(y_test, rule_preds, rule_probs, rule_lats),
        "Hybrid_Weighted": evaluate(y_test, fuse_preds, fuse_probs, fuse_lats)
    }
    
    report_df = pd.DataFrame(metrics).T
    print("\n=== Final Cross-Dataset Evaluation ===")
    print(report_df)
    
    os.makedirs("experiments/reports", exist_ok=True)
    with open("experiments/reports/final_evaluation_report.md", "w") as f:
        f.write("# PhishGuard Final Evaluation Report\n\n")
        f.write("## 1. Cross-Dataset Robustness (Hypothesis 5)\n")
        f.write("This evaluation measures performance on the adversarial cross-dataset split (domains grouped specifically to test generalization).\n\n")
        f.write(report_df.to_markdown())
        f.write("\n\n## 2. Analysis\n")
        f.write("Based on the results, we can observe how the base ML model (XGBoost), the deterministic Rule Engine, and the Hybrid Fusion approach handle completely unseen domains. ")
        
    with open("experiments/reports/false_positives.json", "w") as f:
        json.dump(fp_diagnostics, f, indent=2)
        
    meta = {
        "experiment_id": "final_evaluation",
        "dataset": "cross_dataset",
        "split": "test",
        "model_type": "xgboost",
        "model_version": model_version,
        "artifact_path": model_path,
        "feature_schema_version": feature_schema_version,
        "fusion_strategy": "weighted_sum",
        "ml_weight": alpha,
        "rule_weight": 1.0 - alpha,
        "t_low": t_low,
        "t_high": t_high,
        "ml_only_threshold": ml_thresh,
        "rule_only_threshold": rule_thresh,
        "random_seed": 42,
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open("experiments/reports/final_evaluation_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
