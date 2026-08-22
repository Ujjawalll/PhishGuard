import ipaddress
import socket
import urllib.parse

def is_safe_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
        # Block loopback, private, multicast, reserved
        if ip.is_loopback or ip.is_private or ip.is_multicast or ip.is_reserved:
            return False
        # Block cloud metadata specifically
        if ip_str == "169.254.169.254":
            return False
        return True
    except ValueError:
        return False

def resolve_and_check_ssrf(hostname: str) -> bool:
    """Resolve a hostname and check if it points to a safe IP."""
    try:
        # Use getaddrinfo to resolve
        addr_info = socket.getaddrinfo(hostname, None)
        for res in addr_info:
            ip = res[4][0]
            if not is_safe_ip(ip):
                return False
        return True
    except socket.gaierror:
        # If it fails to resolve, we can't connect anyway, but it's not an SSRF risk
        return True

def validate_url(url: str) -> bool:
    """Check basic URL validity before fetching."""
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        if not parsed.hostname:
            return False
        return resolve_and_check_ssrf(parsed.hostname)
    except Exception:
        return False
