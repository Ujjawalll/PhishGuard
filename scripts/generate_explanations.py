import os
import joblib
import pandas as pd
import shap
import json
import matplotlib.pyplot as plt
from glob import glob
from ml.explainability.explainer import Explainer
from ml.rules.engine import RuleEngine

FEAT_DIR = "ml/data/features/random"

def get_latest_model(model_prefix):
    paths = glob(f"ml/models/{model_prefix}_*")
    return sorted(paths)[-1]

def main():
    print("Loading XGBoost Model...")
    xgb_path = get_latest_model("xgboost")
    pipeline = joblib.load(os.path.join(xgb_path, "model.joblib"))
    
    val_df = pd.read_csv(os.path.join(FEAT_DIR, "val_features.csv"))
    feature_cols = [c for c in val_df.columns if c not in ['label', 'domain', 'url']]
    
    explainer = Explainer(pipeline, feature_cols)
    rule_engine = RuleEngine()
    
    print("Generating Global Feature Importance Plot...")
    X_val = val_df[feature_cols]
    X_val_scaled = pipeline.named_steps['scaler'].transform(X_val)
    shap_values = explainer.explainer.shap_values(X_val_scaled)
    
    os.makedirs("experiments/reports", exist_ok=True)
    # SHAP Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
    plt.savefig("experiments/reports/shap_global_importance.png", bbox_inches="tight")
    print("Saved global importance plot to experiments/reports/shap_global_importance.png")
    
    # Generate examples
    print("\nGenerating Example Explanations...")
    # Pick one phishing and one legitimate
    phish_row = val_df[val_df['label'] == 1].iloc[0]
    legit_row = val_df[val_df['label'] == 0].iloc[0]
    
    examples = []
    
    for row in [legit_row, phish_row]:
        features_dict = row[feature_cols].to_dict()
        url = row['url']
        
        # 1. Rule Engine evaluation
        rule_res = rule_engine.evaluate(url, features_dict)
        
        # 2. ML Probability
        df_single = pd.DataFrame([features_dict])
        ml_prob = pipeline.predict_proba(df_single)[0][1]
        
        # 3. Simple OR Fusion Score
        rule_score = rule_res["normalized_score"]
        fused_score = max(ml_prob, rule_score) # simplified fusion for example
        
        # Risk level determination
        risk = "HIGH_RISK" if fused_score >= 0.5 else "SAFE"
        
        # 4. Generate Explanations
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
        
    with open("experiments/reports/explanation_examples.json", "w") as f:
        json.dump(examples, f, indent=2)
        
    print("Saved explanation examples to experiments/reports/explanation_examples.json")

if __name__ == "__main__":
    main()
