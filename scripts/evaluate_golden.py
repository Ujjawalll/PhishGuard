import pandas as pd
import numpy as np
import os
import joblib
import json
from ml.rules.engine import RuleEngine
from ml.fusion.strategies import WeightedSumFusion
from ml.features.lexical import extract_lexical_features

def main():
    print("Loading production config...")
    with open("configs/production.json") as f:
        config = json.load(f)
        
    xgb_path = config["model"]["artifact_path"]
    t_low = config["risk_thresholds"]["low_to_suspicious"]
    t_high = config["risk_thresholds"]["suspicious_to_high"]
    alpha = config["fusion"]["ml_weight"]
    
    with open(os.path.join(xgb_path, "metadata.json")) as f:
        feature_cols = json.load(f)["features"]
        
    print(f"Loaded ML model from: {xgb_path}")
    ml_model = joblib.load(os.path.join(xgb_path, "model.joblib"))
    rule_engine = RuleEngine()
    fusion = WeightedSumFusion(alpha=alpha, threshold=t_high)
    
    # Golden dataset URLs
    golden_urls = [
        {"url": "https://github.com", "label": 0},
        {"url": "https://github.com/login", "label": 0},
        {"url": "https://google.com", "label": 0},
        {"url": "https://youtube.com", "label": 0},
        {"url": "https://wikipedia.org", "label": 0},
        {"url": "https://chase.com/secure/login", "label": 0},
        {"url": "https://mit.edu", "label": 0},
        {"url": "https://a-very-long-and-hyphenated-but-safe-domain-name.com/some/query?q=123", "label": 0},
        {"url": "http://sub1.sub2.sub3.legit.com", "label": 0},
        {"url": "http://secure-login-verify-account.bad-phishing-site.xyz/login.php", "label": 1},
        {"url": "http://192.168.1.1/admin", "label": 1},
    ]
    
    results = []
    
    print("Evaluating Golden Dataset...")
    for item in golden_urls:
        url = item["url"]
        label = item["label"]
        features = extract_lexical_features(url)
        
        # Rule prediction
        res = rule_engine.evaluate(url, features)
        rule_score = res["normalized_score"]
        
        # ML prediction
        df_single = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}])
        ml_prob = ml_model.predict_proba(df_single)[0][1]
        
        # Fusion prediction
        fused_score = float(fusion.predict_proba(np.array([ml_prob]), np.array([rule_score]))[0])
        
        if fused_score >= t_high:
            final_class = "HIGH_RISK"
        elif fused_score >= t_low:
            final_class = "SUSPICIOUS"
        else:
            final_class = "LOW_RISK"
            
        results.append({
            "url": url,
            "label": label,
            "rule_score": rule_score,
            "ml_probability": ml_prob,
            "fused_score": fused_score,
            "final_class": final_class
        })
        
    report_df = pd.DataFrame(results)
    print("\n=== Golden Dataset Report ===")
    print(report_df.to_string())
    
    os.makedirs("experiments/reports", exist_ok=True)
    report_df.to_csv("experiments/reports/golden_evaluation.csv", index=False)
    print("\nSaved to experiments/reports/golden_evaluation.csv")

if __name__ == "__main__":
    main()
