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
        self.categories = self.config.get("categories", {})
        self.max_score = self.config.get("max_possible_score", 100.0)
        
        # High-risk TLDs
        self.suspicious_tlds = {"tk", "ml", "ga", "cf", "gq"}

    def evaluate(self, url: str, features: Dict[str, Any]) -> Dict[str, Any]:
        triggered_rules = []
        
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
        if features.get("url_length", 0) > 100:
            length = features["url_length"]
            triggered_rules.append(self._build_result("R_LONG_URL", evidence=f"Length is {length} > 100"))
            
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

        # DEEP FEATURES
        # R_DOMAIN_AGE
        age = features.get("domain_age_days")
        if age is not None and 0 <= age < 30:
            triggered_rules.append(self._build_result("R_DOMAIN_AGE", evidence=f"Domain is {age} days old"))
            
        # R_WHOIS_PRIVACY
        if features.get("whois_privacy") == 1:
            triggered_rules.append(self._build_result("R_WHOIS_PRIVACY", evidence="WHOIS privacy protection enabled"))
            
        # R_CERT_INVALID
        if "cert_valid" in features and features["cert_valid"] == 0:
            # only trigger if it's HTTPS
            if parsed.scheme == "https":
                triggered_rules.append(self._build_result("R_CERT_INVALID", evidence="Certificate invalid or expired"))
                
        # R_CERT_MISMATCH
        if "cert_hostname_match" in features and features["cert_hostname_match"] == 0:
            if parsed.scheme == "https":
                triggered_rules.append(self._build_result("R_CERT_MISMATCH", evidence="Certificate hostname does not match URL"))
                
        # HTML DOM
        if features.get("password_input_count", 0) > 0:
            triggered_rules.append(self._build_result("R_PASSWORD_INPUT", evidence=f"Found {features['password_input_count']} password inputs"))
            
        if features.get("external_link_ratio", 0) > 0.8:
            triggered_rules.append(self._build_result("R_EXTERNAL_LINKS", evidence=f"External links: {features['external_link_ratio']*100:.1f}%"))
            
        if features.get("has_redirect") == 1:
            triggered_rules.append(self._build_result("R_META_REDIRECT", evidence="Found meta refresh redirect"))
            
        # Changed text token count key to avoid conflict with url suspicious tokens
        # The worker sets 'suspicious_text_count'
        if features.get("suspicious_text_count", 0) > 3:
            triggered_rules.append(self._build_result("R_SUSPICIOUS_CONTENT", evidence=f"Found {features['suspicious_text_count']} suspicious words in HTML"))

        # Calculate scores per category with caps
        category_scores = {cat: 0.0 for cat in self.categories}
        
        for r in triggered_rules:
            cat = r.get("category", "UNKNOWN")
            if cat in category_scores:
                category_scores[cat] += r["score"]
            else:
                category_scores[cat] = r["score"]
                
        # Apply caps
        final_category_scores = {}
        raw_score = 0.0
        for cat, score in category_scores.items():
            cap = self.categories.get(cat, {}).get("max", float("inf"))
            capped_score = min(score, cap)
            final_category_scores[cat] = capped_score
            raw_score += capped_score
            
        normalized_score = min(raw_score / self.max_score, 1.0)

        return {
            "triggered_rules": triggered_rules,
            "raw_score": raw_score,
            "normalized_score": normalized_score,
            "category_scores": final_category_scores,
            "user_explanation": {
                "top_reasons": [r["description"] for r in sorted(triggered_rules, key=lambda x: x["score"], reverse=True)[:3]]
            },
            "rule_config_version": self.config.get("version", "1.0")
        }

    def _build_result(self, rule_id: str, evidence: str) -> Dict[str, Any]:
        meta = self.rules_meta.get(rule_id, {"weight": 0.0, "category": "UNKNOWN", "description": ""})
        return {
            "rule_id": rule_id,
            "triggered": True,
            "score": meta["weight"],
            "category": meta["category"],
            "description": meta["description"],
            "evidence": evidence
        }
