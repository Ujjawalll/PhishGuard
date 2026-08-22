import whois
from datetime import datetime
from typing import Dict, Any

def extract_whois(domain: str) -> Dict[str, Any]:
    result = {
        "domain_age_days": -1,
        "domain_update_age_days": -1,
        "domain_expiry_days": -1,
        "registrar": "",
        "whois_privacy": 0
    }
    
    if not domain:
        return result
        
    try:
        w = whois.whois(domain)
        now = datetime.now()
        
        # Helper to handle list of dates returned by python-whois
        def get_first_date(date_obj):
            if isinstance(date_obj, list):
                return date_obj[0]
            return date_obj

        creation = get_first_date(w.creation_date)
        if creation and isinstance(creation, datetime):
            result["domain_age_days"] = (now - creation).days
            
        update = get_first_date(w.updated_date)
        if update and isinstance(update, datetime):
            result["domain_update_age_days"] = (now - update).days
            
        expiry = get_first_date(w.expiration_date)
        if expiry and isinstance(expiry, datetime):
            result["domain_expiry_days"] = (expiry - now).days
            
        if w.registrar:
            result["registrar"] = str(w.registrar).lower()
            
        # Very basic privacy check
        privacy_keywords = ['privacy', 'proxy', 'protect', 'redacted', 'hidden', 'guardian', 'withheld']
        if w.registrar and any(k in str(w.registrar).lower() for k in privacy_keywords):
            result["whois_privacy"] = 1
        if w.emails and any(k in str(w.emails).lower() for k in privacy_keywords):
            result["whois_privacy"] = 1
            
    except Exception:
        pass
        
    return result
