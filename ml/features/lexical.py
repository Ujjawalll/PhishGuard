import urllib.parse
import math
import re
import tldextract
import ipaddress
from typing import Dict, Any

SUSPICIOUS_TOKENS = {'login', 'secure', 'account', 'verify', 'update', 'bank', 'confirm', 'suspended', 'support'}
SPECIAL_CHARS = set('@~!$&*+=_')

def extract_lexical_features(url: str) -> Dict[str, Any]:
    # Normalizing URL somewhat for robust feature extraction
    url_lower = url.lower()
    
    # URL parsing
    try:
        parsed = urllib.parse.urlparse(url_lower)
        netloc = parsed.netloc
        path = parsed.path
        query = parsed.query
    except Exception:
        # Fallback for completely malformed URLs
        parsed = None
        netloc = ""
        path = ""
        query = ""

    ext = tldextract.extract(url_lower)

    # Calculate entropy
    def calculate_entropy(s: str) -> float:
        if not s:
            return 0.0
        prob = [s.count(c) / len(s) for c in sorted(set(s))]
        return -sum(p * math.log2(p) for p in prob)

    # Check for IP hostname
    def is_ip_address(host: str) -> bool:
        # Host might have port like 127.0.0.1:8000
        host_no_port = host.split(':')[0] if ':' in host else host
        # Also tldextract gives empty suffix if IP
        try:
            ipaddress.ip_address(host_no_port)
            return True
        except ValueError:
            return False

    # Digit / Letter count
    digits = sum(c.isdigit() for c in url_lower)
    letters = sum(c.isalpha() for c in url_lower)

    # Token counting for suspicious words
    # Split by non-alphanumeric
    tokens = set(re.split(r'\W+', url_lower))
    suspicious_count = sum(1 for token in tokens if token in SUSPICIOUS_TOKENS)
    
    # Path token count
    path_tokens = [p for p in path.split('/') if p] if path else []
    
    # Query param count
    query_params = urllib.parse.parse_qs(query) if query else {}
    
    # Subdomain count
    subdomains = [s for s in ext.subdomain.split('.') if s] if ext.subdomain else []

    return {
        # "url_length": len(url),
        "hostname_length": len(netloc),
        # "path_length": len(path),
        # "query_length": len(query),
        "dot_count": url_lower.count('.'),
        "subdomain_count": len(subdomains),
        "hyphen_count": netloc.count('-'),
        "digit_count": digits,
        "special_char_count": sum(c in SPECIAL_CHARS for c in url_lower),
        "has_at_symbol": '@' in url_lower,
        "has_ip_hostname": is_ip_address(netloc) if netloc else False,
        "suspicious_token_count": suspicious_count,
        "url_entropy": round(calculate_entropy(url_lower), 4),
        "has_encoding": bool(re.search(r'%[0-9a-f]{2}', url_lower)),
        "has_punycode": 'xn--' in url_lower,
        "digit_letter_ratio": digits / (letters + 1),
        "path_token_count": len(path_tokens),
        "query_param_count": len(query_params)
    }
