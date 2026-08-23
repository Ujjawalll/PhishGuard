import pandas as pd
import numpy as np
import os
import joblib
from glob import glob
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix

from ml.features.lexical import extract_lexical_features
from ml.rules.engine import RuleEngine
from ml.fusion.strategies import WeightedSumFusion

def extract_features(df):
    features_list = []
    for url in tqdm(df['url']):
        features_list.append(extract_lexical_features(url))
    return pd.DataFrame(features_list)

def get_latest_model():
    paths = glob("ml/models/xgboost_*")
    return sorted(paths)[-1]

def evaluate(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = cm[0][0], 0, 0, 0 # Fallback
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "fpr": fpr,
        "cm_tn": tn,
        "cm_fp": fp,
        "cm_fn": fn,
        "cm_tp": tp
    }

def main():
    print("Loading test data...")
    test_df = pd.read_csv("ml/data/splits/cross_dataset/test.csv")
    y_test = test_df['label'].values
    
    print("Extracting features...")
    X_test = extract_features(test_df)
    
    print("Loading model and rules...")
    model_path = get_latest_model()
    pipeline = joblib.load(os.path.join(model_path, "model.joblib"))
    
    import json
    with open(os.path.join(model_path, "metadata.json")) as f:
        feature_cols = json.load(f)["features"]
        
    X_test_aligned = X_test[feature_cols]
    
    rule_engine = RuleEngine()
    
    print("Generating predictions...")
    ml_prob = pipeline.predict_proba(X_test_aligned)[:, 1]
    ml_pred = (ml_prob >= 0.22).astype(int) # Best threshold from tuning
    
    rule_scores = []
    features_dicts = X_test.to_dict('records')
    for url, features in zip(test_df['url'], features_dicts):
        res = rule_engine.evaluate(url, features)
        rule_scores.append(res['normalized_score'])
        
    rule_scores = np.array(rule_scores)
    rule_thresh = 0.20 # Based on tuning
    rule_pred = (rule_scores >= rule_thresh).astype(int)
    
    fusion = WeightedSumFusion(alpha=0.6, threshold=0.20)
    fuse_prob = fusion.predict_proba(ml_prob, rule_scores)
    fuse_pred = (fuse_prob >= 0.20).astype(int)
    
    print("\nCalculating metrics...")
    metrics = {
        "ML_Only": evaluate(y_test, ml_pred, ml_prob),
        "Rules_Only": evaluate(y_test, rule_pred, rule_scores),
        "Hybrid_Weighted": evaluate(y_test, fuse_pred, fuse_prob)
    }
    
    report_df = pd.DataFrame(metrics).T
    print("\n=== Final Cross-Dataset Evaluation ===")
    print(report_df)
    
    # Save markdown report
    os.makedirs("experiments/reports", exist_ok=True)
    with open("experiments/reports/final_evaluation_report.md", "w") as f:
        f.write("# PhishGuard Final Evaluation Report\n\n")
        f.write("## 1. Cross-Dataset Robustness (Hypothesis 5)\n")
        f.write("This evaluation measures performance on the adversarial cross-dataset split (domains grouped specifically to test generalization).\n\n")
        f.write(report_df.to_markdown())
        f.write("\n\n## 2. Analysis\n")
        f.write("Based on the results, we can observe how the base ML model (XGBoost), the deterministic Rule Engine, and the Hybrid Fusion approach handle completely unseen domains. ")

if __name__ == "__main__":
    main()
