import pytest
import os
import json
import joblib
from glob import glob
from ml.features.schema import CURRENT_FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA
from ml.rules.engine import RuleEngine
from ml.fusion.strategies import WeightedSumFusion
from backend.app.schemas.scan import ScanResult
from pydantic import ValidationError

def test_risk_enum_validation():
    from datetime import datetime
    # Should pass
    ScanResult(
        scan_id="123", url="http://a.com", domain="a.com", risk_level="LOW_RISK",
        ml_probability=0.1, rule_score=0.1, fused_score=0.1, stage="fast",
        triggered_rules=[], explanation={"risk_level":"LOW_RISK", "top_reasons":[], "recommendation":""},
        model_version="1", feature_schema_version="1", rule_config_version="1",
        scan_timestamp=datetime.utcnow(), deep_analysis_available=False, metadata_failures=[]
    )
    
    # Should fail due to SAFE (which is removed)
    with pytest.raises(ValidationError):
        ScanResult(
            scan_id="123", url="http://a.com", domain="a.com", risk_level="SAFE",
            ml_probability=0.1, rule_score=0.1, fused_score=0.1, stage="fast",
            triggered_rules=[], explanation={"risk_level":"SAFE", "top_reasons":[], "recommendation":""},
            model_version="1", feature_schema_version="1", rule_config_version="1",
            scan_timestamp=datetime.utcnow(), deep_analysis_available=False, metadata_failures=[]
        )

def test_score_range_validation():
    rule_engine = RuleEngine()
    res = rule_engine.evaluate("http://example.com", {"url_length": 500, "suspicious_token_count": 100})
    
    assert 0 <= res["normalized_score"] <= 1.0, "Rule score must be bounded [0, 1]"
    
    with open("configs/production.json") as f:
        config = json.load(f)
    alpha = config["fusion"]["ml_weight"]
    t_high = config["risk_thresholds"]["suspicious_to_high"]
    
    fusion = WeightedSumFusion(alpha=alpha, threshold=t_high)
    import numpy as np
    fused = fusion.predict_proba(np.array([0.9]), np.array([0.9]))[0]
    assert 0 <= fused <= 1.0, "Fused score must be bounded [0, 1]"
    
def test_feature_schema_parity():
    paths = glob("ml/models/xgboost_*")
    latest_path = sorted(paths)[-1]
    
    with open(os.path.join(latest_path, "metadata.json")) as f:
        metadata = json.load(f)
        
    training_features = metadata["features"]
    
    assert training_features == FEATURE_SCHEMA, "Training schema must match inference schema exactly"
    
    with open("configs/production.json") as f:
        config = json.load(f)
    alpha = config["fusion"]["ml_weight"]
    t_high = config["risk_thresholds"]["suspicious_to_high"]
    
    fusion = WeightedSumFusion(alpha=alpha, threshold=t_high)
    import numpy as np
    
    fused_normal = fusion.predict_proba(np.array([0.1]), np.array([0.1]), known_malicious=np.array([False]))[0]
    assert fused_normal < t_high
    
    fused_malicious = fusion.predict_proba(np.array([0.1]), np.array([0.1]), known_malicious=np.array([True]))[0]
    assert fused_malicious == 1.0
