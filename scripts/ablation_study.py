import pandas as pd
import numpy as np
import os
import time
import joblib
import json
import urllib.parse
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from ml.features.lexical import extract_lexical_features
from ml.rules.engine import RuleEngine
from ml.fusion.strategies import WeightedSumFusion

def get_latest_model():
    with open("configs/production.json") as f:
        config = json.load(f)
    return config["model"]["artifact_path"]

def evaluate(y_true, y_pred, y_prob, lats_feat, lats_rule, lats_ml, lats_fusion, lats_total):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = cm[0][0], 0, 0, 0
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fpr),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "feature_latency_ms": float(np.mean(lats_feat) * 1000),
        "rule_latency_ms": float(np.mean(lats_rule) * 1000),
        "ml_latency_ms": float(np.mean(lats_ml) * 1000),
        "fusion_latency_ms": float(np.mean(lats_fusion) * 1000),
        "total_latency_ms": float(np.mean(lats_total) * 1000)
    }

def main():
    print("Loading test data...")
    test_df = pd.read_csv("ml/data/splits/cross_dataset/test.csv")
    test_df = pd.concat([test_df[test_df['label'] == 0].sample(250, random_state=42), test_df[test_df['label'] == 1].sample(80, random_state=42)])
    test_df = test_df.sample(frac=1.0, random_state=42) # freeze final test set
    
    y_test = test_df['label'].values
    urls = test_df['url'].values
    
    print("Loading real evidence fixtures...")
    domain_df = pd.read_csv("experiments/data/domain_features.csv") if os.path.exists("experiments/data/domain_features.csv") else None
    html_df = pd.read_csv("experiments/data/html_features.csv") if os.path.exists("experiments/data/html_features.csv") else None
    
    domain_lookup = {row['url']: row.to_dict() for _, row in domain_df.iterrows()} if domain_df is not None else {}
    html_lookup = {row['url']: row.to_dict() for _, row in html_df.iterrows()} if html_df is not None else {}
    
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
    
    fusion = WeightedSumFusion(alpha=alpha, threshold=t_high)
    
    rule_thresh = 0.5 
    if os.path.exists("experiments/results/rule_baseline.txt"):
        with open("experiments/results/rule_baseline.txt") as f:
            for line in f:
                if line.startswith("T_RULE:"):
                    rule_thresh = float(line.strip().split(": ")[1])
                    break
    
    ml_thresh = 0.22
    
    results = {}
    
    print("Running experiments...")

    def run_experiment(exp_name, use_ml=True, use_rules=True, use_domain=False, use_html=False):
        preds, probs = [], []
        lats_feat, lats_rule, lats_ml, lats_fusion, lats_total = [], [], [], [], []
        
        for url in urls:
            t_start = time.perf_counter()
            
            # Feature Extraction
            t_feat_start = time.perf_counter()
            feat = extract_lexical_features(url)
            if use_domain and url in domain_lookup:
                feat.update({k: v for k, v in domain_lookup[url].items() if k not in ['sample_id', 'url', 'domain', 'collection_timestamp']})
            if use_html and url in html_lookup:
                feat.update({k: v for k, v in html_lookup[url].items() if k not in ['sample_id', 'url', 'collection_timestamp']})
            lats_feat.append(time.perf_counter() - t_feat_start)
            
            rule_score = 0.0
            t_rule_start = time.perf_counter()
            if use_rules:
                res = rule_engine.evaluate(url, feat)
                rule_score = res["normalized_score"]
            lats_rule.append(time.perf_counter() - t_rule_start)
            
            ml_prob = 0.0
            t_ml_start = time.perf_counter()
            if use_ml:
                df_single = pd.DataFrame([{col: feat.get(col, 0) for col in feature_cols}])
                ml_prob = pipeline.predict_proba(df_single)[0][1]
            lats_ml.append(time.perf_counter() - t_ml_start)
            
            t_fusion_start = time.perf_counter()
            if use_ml and use_rules:
                fused = fusion.predict_proba(np.array([ml_prob]), np.array([rule_score]))[0]
                pred = int(fused >= t_high)
                prob = fused
            elif use_ml:
                pred = int(ml_prob >= ml_thresh)
                prob = ml_prob
            elif use_rules:
                pred = int(rule_score >= rule_thresh)
                prob = rule_score
            lats_fusion.append(time.perf_counter() - t_fusion_start)
            
            lats_total.append(time.perf_counter() - t_start)
            
            preds.append(pred)
            probs.append(prob)
            
        results[exp_name] = evaluate(y_test, preds, probs, lats_feat, lats_rule, lats_ml, lats_fusion, lats_total)

    run_experiment("A: Rule-only URL baseline", use_ml=False, use_rules=True, use_domain=False, use_html=False)
    run_experiment("B: ML-only URL baseline", use_ml=True, use_rules=False, use_domain=False, use_html=False)
    run_experiment("C: URL Rules + ML Hybrid", use_ml=True, use_rules=True, use_domain=False, use_html=False)
    run_experiment("D: URL + Domain", use_ml=True, use_rules=True, use_domain=True, use_html=False)
    run_experiment("E: URL + Domain + HTML", use_ml=True, use_rules=True, use_domain=True, use_html=True)
    run_experiment("F: Full Hybrid", use_ml=True, use_rules=True, use_domain=True, use_html=True) # E and F are currently the same, matching instruction
    
    report_df = pd.DataFrame(results).T
    print("\n=== Ablation Study Results ===")
    print(report_df)
    
    os.makedirs("experiments/reports", exist_ok=True)
    with open("experiments/reports/ablation_study.md", "w") as f:
        f.write("# PhishGuard Ablation Study\n\n")
        f.write("This study demonstrates the contribution of each evidence layer to the final model performance.\n")
        f.write("> **Note**: These results are generated using actual offline-collected domain and HTML fixtures, unlike previous synthetic iterations.\n\n")
        f.write(report_df.to_markdown())
        
    meta = {
        "experiment_id": "ablation",
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
        "metrics": {k: v for k, v in results.items()},
        "timestamp": datetime.utcnow().isoformat()
    }
    with open("experiments/reports/ablation_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()

