import pandas as pd
import json
import joblib
from glob import glob
from ml.rules.engine import RuleEngine
from ml.explainability.explainer import Explainer
from ml.fusion.strategies import WeightedSumFusion
from ml.features.lexical import extract_lexical_features

golden_data = [
    {"url": "https://www.google.com/search?q=test&hl=en", "label": 0},
    {"url": "https://github.com/login?return_to=https%3A%2F%2Fgithub.com%2Fdashboard", "label": 0},
    {"url": "https://secure.bankofamerica.com/login/sign-in/signOnV2Screen.go", "label": 0},
    {"url": "http://192.168.1.1/admin/login.php", "label": 1},
    {"url": "https://paypal-update-account.secure-verify-info.com/login", "label": 1},
    {"url": "https://xn--80ak6aa92e.com/secure", "label": 1} # apple.com in punycode
]

def main():
    paths = glob("ml/models/xgboost_*")
    latest_path = sorted(paths)[-1]
    pipeline = joblib.load(latest_path + "/model.joblib")
    
    with open(latest_path + "/metadata.json", "r") as f:
        feature_cols = json.load(f)["features"]
        
    rule_engine = RuleEngine()
    fusion = WeightedSumFusion(alpha=0.6, threshold=0.6)
    
    results = []
    for item in golden_data:
        url = item["url"]
        features = extract_lexical_features(url)
        rule_res = rule_engine.evaluate(url, features)
        
        df_single = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}])
        ml_prob = pipeline.predict_proba(df_single)[0][1]
        
        fused_score = fusion.predict_proba(
            __import__('numpy').array([ml_prob]), 
            __import__('numpy').array([rule_res["normalized_score"]]),
            known_malicious=__import__('numpy').array([features.get("is_known_malicious", False)])
        )[0]
        
        if fused_score >= 0.20:
            risk = "HIGH_RISK"
        elif fused_score >= 0.08:
            risk = "SUSPICIOUS"
        else:
            risk = "LOW_RISK"
            
        results.append({
            "url": url,
            "label": item["label"],
            "features": features,
            "rule_score": rule_res["normalized_score"],
            "ml_probability": ml_prob,
            "fused_score": fused_score,
            "risk": risk
        })
        
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
