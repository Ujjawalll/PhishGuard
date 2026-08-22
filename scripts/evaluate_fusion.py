import pandas as pd
import numpy as np
import os
import joblib
from glob import glob
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from ml.rules.engine import RuleEngine
from ml.fusion.strategies import WeightedSumFusion, MetaClassifierFusion, OrLogicFusion, HierarchicalFusion

FEAT_DIR = "ml/data/features/random"

def get_latest_model(model_prefix):
    paths = glob(f"ml/models/{model_prefix}_*")
    if not paths:
        raise ValueError(f"No models found for {model_prefix}")
    # Sort by timestamp, get latest
    return sorted(paths)[-1]

def get_predictions(df, feature_cols, ml_model, rule_engine):
    print("Generating base predictions...")
    
    # ML Predictions
    X = df[feature_cols]
    ml_prob = ml_model.predict_proba(X)[:, 1]
    
    # Rule Predictions
    rule_scores = []
    # Reconstruct dictionary from feature columns
    features_dicts = X.to_dict('records')
    for url, features in zip(df['url'], features_dicts):
        res = rule_engine.evaluate(url, features)
        rule_scores.append(res['normalized_score'])
        
    return np.array(ml_prob), np.array(rule_scores), df['label'].values

def evaluate(y_true, y_pred, y_prob):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

def main():
    print("Loading Validation Data...")
    train_df = pd.read_csv(os.path.join(FEAT_DIR, "train_features.csv"))
    val_df = pd.read_csv(os.path.join(FEAT_DIR, "val_features.csv"))
    
    feature_cols = [c for c in val_df.columns if c not in ['label', 'domain', 'url']]
    
    # Load XGBoost model
    xgb_path = get_latest_model("xgboost")
    print(f"Loaded ML model from: {xgb_path}")
    ml_model = joblib.load(os.path.join(xgb_path, "model.joblib"))
    
    # Load Rule Engine
    rule_engine = RuleEngine()
    
    # Get base predictions for Train (to fit meta-classifier)
    train_ml, train_rules, y_train = get_predictions(train_df, feature_cols, ml_model, rule_engine)
    
    # Get base predictions for Validation (to evaluate)
    val_ml, val_rules, y_val = get_predictions(val_df, feature_cols, ml_model, rule_engine)
    
    results = {}
    
    print("\nEvaluating Fusion Strategies...")
    
    # 1. Base ML Only
    preds_ml = (val_ml >= 0.5).astype(int)
    results["ML_Only (XGB)"] = evaluate(y_val, preds_ml, val_ml)
    
    # 2. Base Rules Only
    # From config.json: suspicious threshold is ~0.34 (5.0 / 14.5)
    rule_thresh = 5.0 / 14.5
    preds_rule = (val_rules >= rule_thresh).astype(int)
    results["Rule_Only"] = evaluate(y_val, preds_rule, val_rules)
    
    # 3. Weighted Sum (alpha=0.6)
    ws = WeightedSumFusion(alpha=0.6)
    results["Hybrid_WeightedSum"] = evaluate(y_val, ws.predict(val_ml, val_rules), ws.predict_proba(val_ml, val_rules))
    
    # 4. Meta-Classifier
    meta = MetaClassifierFusion().fit(train_ml, train_rules, y_train)
    results["Hybrid_MetaClassifier"] = evaluate(y_val, meta.predict(val_ml, val_rules), meta.predict_proba(val_ml, val_rules))
    
    # 5. OR Logic
    or_logic = OrLogicFusion(ml_threshold=0.5, rule_threshold=rule_thresh)
    results["Hybrid_OR_Logic"] = evaluate(y_val, or_logic.predict(val_ml, val_rules), or_logic.predict_proba(val_ml, val_rules))
    
    # 6. Hierarchical Logic
    hier = HierarchicalFusion(rule_safe_max=2.0/14.5, rule_high_min=5.0/14.5)
    results["Hybrid_Hierarchical"] = evaluate(y_val, hier.predict(val_ml, val_rules), hier.predict_proba(val_ml, val_rules))
    
    # Format and save
    report_df = pd.DataFrame(results).T
    print("\n=== Fusion Comparison Report ===")
    print(report_df)
    
    os.makedirs("experiments/reports", exist_ok=True)
    report_df.to_csv("experiments/reports/fusion_comparison.csv")
    print("\nSaved to experiments/reports/fusion_comparison.csv")

if __name__ == "__main__":
    main()
