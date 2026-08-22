import dns.resolver
import dns.exception

def extract_dns(domain: str) -> dict:
    result = {
        "has_dns_a": 0,
        "has_dns_aaaa": 0,
        "has_mx": 0,
        "ns_count": 0,
        "dns_record_count": 0
    }
    
    if not domain:
        return result
        
    resolver = dns.resolver.Resolver()
    resolver.timeout = 2
    resolver.lifetime = 2
    
    total = 0
    try:
        a_recs = resolver.resolve(domain, 'A')
        result["has_dns_a"] = 1
        total += len(a_recs)
    except dns.exception.DNSException:
        pass
        
    try:
        aaaa_recs = resolver.resolve(domain, 'AAAA')
        result["has_dns_aaaa"] = 1
        total += len(aaaa_recs)
    except dns.exception.DNSException:
        pass
        
    try:
        mx_recs = resolver.resolve(domain, 'MX')
        result["has_mx"] = 1
        total += len(mx_recs)
    except dns.exception.DNSException:
        pass
        
    try:
        ns_recs = resolver.resolve(domain, 'NS')
        result["ns_count"] = len(ns_recs)
        total += len(ns_recs)
    except dns.exception.DNSException:
        pass
        
    result["dns_record_count"] = total
    return result
