import ssl
import socket
from datetime import datetime

def extract_tls(hostname: str) -> dict:
    result = {
        "cert_valid": 0,
        "cert_issuer": "",
        "cert_age_days": -1,
        "san_count": 0,
        "cert_hostname_match": 0
    }
    
    if not hostname:
        return result
        
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE  # We just want to fetch it, validity check is manual
    
    try:
        with socket.create_connection((hostname, 443), timeout=3) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert(binary_form=False)
                if not cert:
                    # Let's try getting it with verification on to see what it is
                    context.verify_mode = ssl.CERT_REQUIRED
                    try:
                        with socket.create_connection((hostname, 443), timeout=3) as sock2:
                            with context.wrap_socket(sock2, server_hostname=hostname) as ssock2:
                                cert = ssock2.getpeercert()
                    except Exception:
                        pass
                
                if cert:
                    # Check validity manually
                    not_before = datetime.strptime(cert['notBefore'], "%b %d %H:%M:%S %Y %Z")
                    not_after = datetime.strptime(cert['notAfter'], "%b %d %H:%M:%S %Y %Z")
                    now = datetime.utcnow()
                    
                    if not_before <= now <= not_after:
                        result["cert_valid"] = 1
                        
                    result["cert_age_days"] = (now - not_before).days
                    
                    issuer = dict(x[0] for x in cert.get('issuer', []))
                    result["cert_issuer"] = issuer.get('organizationName', '')
                    
                    san = cert.get('subjectAltName', [])
                    result["san_count"] = len(san)
                    
                    # Match hostname
                    hostnames = [x[1] for x in san if x[0] == 'DNS']
                    if any(hostname == h or (h.startswith('*.') and hostname.endswith(h[2:])) for h in hostnames):
                        result["cert_hostname_match"] = 1
                    
    except Exception:
        pass
        
    return result
