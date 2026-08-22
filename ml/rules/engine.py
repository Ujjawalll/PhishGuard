import json
import os
from typing import Dict, Any, List
import urllib.parse
import tldextract

class RuleEngine:
    def __init__(self, config_path: str = None):
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.rules_meta = {r["rule_id"]: r for r in self.config["rules"]}
        self.max_score = self.config.get("max_possible_score", 15.0)
        
        # High-risk TLDs
        self.suspicious_tlds = {"tk", "ml", "ga", "cf", "gq"}

    def evaluate(self, url: str, features: Dict[str, Any]) -> Dict[str, Any]:
        triggered_rules = []
        raw_score = 0.0
        
        parsed = urllib.parse.urlparse(url.lower())
        ext = tldextract.extract(url.lower())

        # R_IP_HOST
        if features.get("has_ip_hostname"):
            triggered_rules.append(self._build_result("R_IP_HOST", evidence=f"Hostname is IP: {parsed.netloc}"))
            
        # R_AT_SYMBOL
        if features.get("has_at_symbol"):
            triggered_rules.append(self._build_result("R_AT_SYMBOL", evidence="Found '@' in URL"))

        # R_BRAND_TOKEN
        if features.get("suspicious_token_count", 0) > 0:
            count = features["suspicious_token_count"]
            triggered_rules.append(self._build_result("R_BRAND_TOKEN", evidence=f"Found {count} suspicious token(s)"))
            
        # R_LONG_URL
        if features.get("url_length", 0) > 75:
            length = features["url_length"]
            triggered_rules.append(self._build_result("R_LONG_URL", evidence=f"Length is {length} > 75"))
            
        # R_DEEP_SUBDOMAINS
        if features.get("subdomain_count", 0) > 3:
            count = features["subdomain_count"]
            triggered_rules.append(self._build_result("R_DEEP_SUBDOMAINS", evidence=f"{count} subdomains > 3"))
            
        # R_SUSPICIOUS_TLD
        if ext.suffix in self.suspicious_tlds:
            triggered_rules.append(self._build_result("R_SUSPICIOUS_TLD", evidence=f"TLD .{ext.suffix} is high-risk"))
            
        # R_NO_HTTPS
        if parsed.scheme == "http":
            triggered_rules.append(self._build_result("R_NO_HTTPS", evidence="Scheme is HTTP"))
            
        # R_PUNYCODE
        if features.get("has_punycode"):
            triggered_rules.append(self._build_result("R_PUNYCODE", evidence="Punycode 'xn--' found in hostname"))

        # Calculate scores
        for r in triggered_rules:
            raw_score += r["score"]
            
        normalized_score = min(raw_score / self.max_score, 1.0)
        
        # Risk levels
        thresholds = self.config["thresholds"]
        if raw_score <= thresholds["safe_max"]:
            risk_level = "SAFE"
        elif raw_score < thresholds["high_risk_min"]:
            risk_level = "SUSPICIOUS"
        else:
            risk_level = "HIGH_RISK"

        # User Explanation
        top_reasons = [r["description"] for r in sorted(triggered_rules, key=lambda x: x["score"], reverse=True)[:3]]
        recommendation = "Proceed with caution." if risk_level == "SUSPICIOUS" else "Do not enter credentials." if risk_level == "HIGH_RISK" else "Appears safe."

        return {
            "triggered_rules": triggered_rules,
            "raw_score": raw_score,
            "normalized_score": normalized_score,
            "risk_level": risk_level,
            "user_explanation": {
                "risk_level": risk_level,
                "top_reasons": top_reasons,
                "recommendation": recommendation
            },
            "rule_config_version": self.config["version"]
        }

    def _build_result(self, rule_id: str, evidence: str) -> Dict[str, Any]:
        meta = self.rules_meta.get(rule_id, {"weight": 0.0, "severity": "Unknown", "description": ""})
        return {
            "rule_id": rule_id,
            "triggered": True,
            "score": meta["weight"],
            "severity": meta["severity"],
            "description": meta["description"],
            "evidence": evidence
        }
