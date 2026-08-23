import os
import joblib
import pandas as pd
import shap
import json
import uuid
import matplotlib.pyplot as plt
from datetime import datetime
from ml.explainability.explainer import Explainer
from ml.rules.engine import RuleEngine
from ml.fusion.strategies import WeightedSumFusion

FEAT_DIR = "ml/data/features/random"

def main():
    print("Loading Config...")
    with open("configs/production.json") as f:
        config = json.load(f)
        
    xgb_path = config["model"]["artifact_path"]
    model_version = config["model"]["production_model"]
    schema_version = config["model"]["feature_schema_version"]
    t_high = config["risk_thresholds"]["suspicious_to_high"]
    t_low = config["risk_thresholds"]["low_to_suspicious"]
    alpha = config["fusion"]["ml_weight"]
    
    pipeline = joblib.load(os.path.join(xgb_path, "model.joblib"))
    
    val_df = pd.read_csv(os.path.join(FEAT_DIR, "val_features.csv"))
    
    with open(os.path.join(xgb_path, "metadata.json")) as f:
        feature_cols = json.load(f)["features"]
    
    explainer = Explainer(pipeline, feature_cols)
    rule_engine = RuleEngine()
    fusion = WeightedSumFusion(alpha=alpha, threshold=t_high)
    
    print("Generating Global Feature Importance Plot...")
    X_val = val_df[feature_cols]
    X_val_scaled = pipeline.named_steps['scaler'].transform(X_val)
    shap_values = explainer.explainer.shap_values(X_val_scaled)
    
    os.makedirs("experiments/reports", exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
    plt.savefig("experiments/reports/shap_global_importance.png", bbox_inches="tight")
    print("Saved global importance plot to experiments/reports/shap_global_importance.png")
    
    print("\nGenerating Example Explanations...")
    phish_row = val_df[val_df['label'] == 1].iloc[0]
    legit_row = val_df[val_df['label'] == 0].iloc[0]
    
    examples = []
    
    for row in [legit_row, phish_row]:
        features_dict = row[feature_cols].to_dict()
        url = row['url']
        
        rule_res = rule_engine.evaluate(url, features_dict)
        df_single = pd.DataFrame([features_dict])
        ml_prob = pipeline.predict_proba(df_single)[0][1]
        
        rule_score = rule_res["normalized_score"]
        fused_score = fusion.predict_proba(pd.Series([ml_prob]), pd.Series([rule_score]))[0]
        
        if fused_score >= t_high:
            risk = "HIGH_RISK"
        elif fused_score >= t_low:
            risk = "SUSPICIOUS"
        else:
            risk = "LOW_RISK"
        
        shap_dict = explainer.get_shap_values(features_dict)
        
        analyst_exp = explainer.build_analyst_explanation(
            features_dict=features_dict,
            ml_prob=ml_prob,
            rule_score=rule_score,
            fused_score=fused_score,
            triggered_rules=rule_res["triggered_rules"],
            metadata={"url": url, "label": "Phishing" if row['label'] == 1 else "Legitimate"}
        )
        
        user_exp = explainer.build_user_explanation(
            risk_level=risk,
            triggered_rules=rule_res["triggered_rules"],
            shap_dict=shap_dict
        )
        
        examples.append({
            "url": url,
            "actual_label": int(row['label']),
            "user_explanation": user_exp,
            "analyst_explanation": analyst_exp
        })
        
    experiment_id = str(uuid.uuid4())
    report = {
        "experiment_id": experiment_id,
        "timestamp": datetime.utcnow().isoformat(),
        "dataset": "random/val_features.csv",
        "split": "val",
        "features": "lexical",
        "model_type": "xgboost",
        "model_version": model_version,
        "artifact_path": xgb_path,
        "feature_schema_version": schema_version,
        "fusion_strategy": "weighted_sum",
        "ML_weight": alpha,
        "rule_weight": 1.0 - alpha,
        "T_LOW": t_low,
        "T_HIGH": t_high,
        "examples": examples
    }
        
    with open("experiments/reports/explanation_examples.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("Saved explanation examples to experiments/reports/explanation_examples.json")

if __name__ == "__main__":
    main()
