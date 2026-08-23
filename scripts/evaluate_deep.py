import pandas as pd
import json
import joblib
from glob import glob
from ml.rules.engine import RuleEngine
from ml.fusion.strategies import WeightedSumFusion
from ml.features.lexical import extract_lexical_features
from worker.main import analyze_url

golden_data = [
    {"url": "https://www.google.com", "label": 0},
    {"url": "https://github.com/login", "label": 0}
    # Let's skip hitting actual phishing URLs live to avoid network issues or triggering security appliances, 
    # but we can mock deep features for a phishing URL
]

def main():
    paths = glob("ml/models/xgboost_*")
    latest_path = sorted(paths)[-1]
    pipeline = joblib.load(latest_path + "/model.joblib")
    
    with open(latest_path + "/metadata.json", "r") as f:
        feature_cols = json.load(f)["features"]
        
    rule_engine = RuleEngine()
    import json
    with open("configs/production.json") as f:
        config = json.load(f)
    alpha = config["fusion"]["ml_weight"]
    t_high = config["risk_thresholds"]["suspicious_to_high"]
    
    fusion = WeightedSumFusion(alpha=alpha, threshold=t_high)
    
    results = []
    
    # Analyze a live safe site
    for item in golden_data:
        url = item["url"]
        lexical = extract_lexical_features(url)
        deep = analyze_url(url)
        
        features = lexical.copy()
        features.update(deep)
        
        rule_res = rule_engine.evaluate(url, features)
        
        df_single = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}])
        ml_prob = pipeline.predict_proba(df_single)[0][1]
        
        fused_score = fusion.predict_proba(
            __import__('numpy').array([ml_prob]), 
            __import__('numpy').array([rule_res["normalized_score"]]),
            known_malicious=__import__('numpy').array([features.get("is_known_malicious", False)])
        )[0]
        
        results.append({
            "url": url,
            "rule_score": rule_res["normalized_score"],
            "fused_score": fused_score,
            "triggered_rules": [r["rule_id"] for r in rule_res["triggered_rules"]]
        })
        
    # Analyze a mocked phishing site
    phish_url = "http://192.168.1.1/admin/login.php"
    lexical = extract_lexical_features(phish_url)
    deep_mock = {
        "domain_age_days": 2, 
        "whois_privacy": 1,
        "cert_valid": 0,
        "password_input_count": 2,
        "external_link_ratio": 0.9,
        "has_redirect": 1
    }
    features = lexical.copy()
    features.update(deep_mock)
    
    rule_res = rule_engine.evaluate(phish_url, features)
    df_single = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}])
    ml_prob = pipeline.predict_proba(df_single)[0][1]
    
    fused_score = fusion.predict_proba(
        __import__('numpy').array([ml_prob]), 
        __import__('numpy').array([rule_res["normalized_score"]]),
        known_malicious=__import__('numpy').array([features.get("is_known_malicious", False)])
    )[0]
    
    results.append({
        "url": phish_url,
        "rule_score": rule_res["normalized_score"],
        "fused_score": fused_score,
        "triggered_rules": [r["rule_id"] for r in rule_res["triggered_rules"]]
    })
    
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
