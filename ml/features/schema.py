"""
Feature Schema Definitions
"""

# Format: v{major}.{minor}
CURRENT_FEATURE_SCHEMA_VERSION = "v1.0"

class FeatureSchema:
    version: str = CURRENT_FEATURE_SCHEMA_VERSION
    # Lexical features
    url_length: int
    hostname_length: int
    path_length: int
    query_length: int
    dot_count: int
    subdomain_count: int
    hyphen_count: int
    digit_count: int
    special_char_count: int
    has_at_symbol: bool
    has_ip_hostname: bool
    suspicious_token_count: int
    url_entropy: float
    has_encoding: bool
    has_punycode: bool
    digit_letter_ratio: float
    path_token_count: int
    query_param_count: int
    
    # Missing Host/DNS, WHOIS, TLS, HTML features to be added in Phase 2
