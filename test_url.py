import urllib.parse
import json
import pandas as pd
import joblib
import os
from ml.features.lexical import extract_lexical_features
from ml.rules.engine import RuleEngine
from ml.fusion.strategies import WeightedSumFusion

def main():
    url = "https://example.com/login?redirect_url=https%3A%2F%2Fexample.com%2Fdashboard&session_id=abcdef1234567890abcdef1234567890&user_id=12345"
    features = extract_lexical_features(url)
    
    with open("configs/production.json") as f:
        config = json.load(f)
    
    model_path = config["model"]["artifact_path"]
    t_high = config["risk_thresholds"]["suspicious_to_high"]
    t_low = config["risk_thresholds"]["low_to_suspicious"]
    alpha = config["fusion"]["ml_weight"]
    
    print(f"Using model: {model_path}")
    pipeline = joblib.load(os.path.join(model_path, "model.joblib"))
    with open(os.path.join(model_path, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    feature_cols = metadata["features"]
    df_single = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}])
    ml_prob = pipeline.predict_proba(df_single)[0][1]
    
    rule_engine = RuleEngine()
    rule_res = rule_engine.evaluate(url, features)
    
    rule_score = rule_res["normalized_score"]
    
    fusion = WeightedSumFusion(alpha=alpha, threshold=t_high)
    fused_score = float(fusion.predict_proba(
        __import__('numpy').array([ml_prob]), 
        __import__('numpy').array([rule_score])
    )[0])
    
    print(f"ML Prob: {ml_prob}")
    print(f"Rule Score: {rule_score}")
    print(f"Fused Score: {fused_score}")
    if fused_score >= t_high:
        print("Risk Level: HIGH_RISK")
    elif fused_score >= t_low:
        print("Risk Level: SUSPICIOUS")
    else:
        print("Risk Level: LOW_RISK")

if __name__ == "__main__":
    main()
