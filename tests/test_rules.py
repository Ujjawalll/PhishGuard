import pytest
from ml.rules.engine import RuleEngine
from ml.features.lexical import extract_lexical_features

@pytest.fixture
def engine():
    return RuleEngine()

def test_safe_url(engine):
    url = "https://www.example.com"
    features = extract_lexical_features(url)
    res = engine.evaluate(url, features)
    assert "normalized_score" in res
    assert res["normalized_score"] == 0.0
    assert len(res["triggered_rules"]) == 0
    assert res["raw_score"] == 0.0

def test_ip_host_and_http(engine):
    url = "http://192.168.1.1/path"
    features = extract_lexical_features(url)
    res = engine.evaluate(url, features)
    rule_ids = [r["rule_id"] for r in res["triggered_rules"]]
    assert "R_IP_HOST" in rule_ids
    assert "R_NO_HTTPS" in rule_ids
    assert res["raw_score"] >= 4.0

def test_punycode_and_suspicious_tld(engine):
    url = "https://xn--e1awd7f.tk"
    features = extract_lexical_features(url)
    res = engine.evaluate(url, features)
    rule_ids = [r["rule_id"] for r in res["triggered_rules"]]
    assert "R_PUNYCODE" in rule_ids
    assert "R_SUSPICIOUS_TLD" in rule_ids

def test_high_risk_threshold(engine):
    # This URL should trigger IP (20.0), HTTP (5.0), and Brand Token ('login' -> 3.0) = 28.0
    url = "http://192.168.1.1/login.php"
    features = extract_lexical_features(url)
    res = engine.evaluate(url, features)
    assert "normalized_score" in res
    assert res["normalized_score"] > 0
    assert len(res["user_explanation"]["top_reasons"]) <= 3
