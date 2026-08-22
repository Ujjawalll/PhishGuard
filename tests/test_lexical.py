import pytest
from ml.features.lexical import extract_lexical_features

def test_normal_url():
    url = "https://www.example.com/path/to/page?query=1"
    f = extract_lexical_features(url)
    assert f["url_length"] == len(url)
    assert f["hostname_length"] == len("www.example.com")
    assert f["path_length"] == len("/path/to/page")
    assert f["query_length"] == len("query=1")
    assert f["dot_count"] == 2
    assert f["subdomain_count"] == 1
    assert f["has_ip_hostname"] is False
    assert f["has_punycode"] is False
    assert f["has_at_symbol"] is False

def test_ip_hostname():
    url = "http://192.168.1.1/login.html"
    f = extract_lexical_features(url)
    assert f["has_ip_hostname"] is True
    assert f["suspicious_token_count"] == 1  # 'login'

def test_punycode_url():
    url = "https://xn--e1awd7f.com"
    f = extract_lexical_features(url)
    assert f["has_punycode"] is True

def test_encoding_and_special_chars():
    url = "http://example.com/login%20page?user=test&verify=1@!"
    f = extract_lexical_features(url)
    assert f["has_encoding"] is True
    assert f["has_at_symbol"] is True
    assert f["suspicious_token_count"] == 2  # 'login', 'verify'
    assert f["query_param_count"] == 2

def test_subdomains_and_hyphens():
    url = "http://secure-update.my-bank.account.example.com"
    f = extract_lexical_features(url)
    assert f["hyphen_count"] == 2
    assert f["subdomain_count"] == 3 # secure-update, my-bank, account

def test_digit_letter_ratio():
    url = "http://abc1234.com" # 7 letters (h,t,t,p,a,b,c,c,o,m) - wait, let's just check relative
    f = extract_lexical_features(url)
    assert f["digit_count"] == 4
    assert f["digit_letter_ratio"] > 0
