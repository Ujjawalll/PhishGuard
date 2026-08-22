import pytest
from worker.fetcher.security import is_safe_ip, validate_url

def test_is_safe_ip():
    assert is_safe_ip("8.8.8.8") == True
    assert is_safe_ip("127.0.0.1") == False
    assert is_safe_ip("192.168.1.100") == False
    assert is_safe_ip("169.254.169.254") == False

def test_validate_url():
    assert validate_url("http://example.com") == True
    assert validate_url("ftp://example.com") == False # unsupported scheme
    assert validate_url("http://127.0.0.1") == False # ssrf
