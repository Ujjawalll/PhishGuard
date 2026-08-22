import argparse
import json
import urllib.parse
import tldextract
import sys

from worker.fetcher.http_client import SafeHTTPClient
from worker.extractors.dns_extractor import extract_dns
from worker.extractors.whois_extractor import extract_whois
from worker.extractors.tls_extractor import extract_tls
from worker.extractors.html_extractor import extract_html

def analyze_url(url: str) -> dict:
    parsed = urllib.parse.urlparse(url)
    ext = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    
    results = {}
    
    # DNS
    results.update(extract_dns(domain))
    
    # WHOIS
    results.update(extract_whois(domain))
    
    # TLS
    if parsed.scheme == "https":
        results.update(extract_tls(parsed.hostname))
    else:
        # Fill defaults
        results.update(extract_tls("")) 
        
    # HTML
    client = SafeHTTPClient()
    html_content, error = client.fetch_html(url)
    
    if error:
        results["html_error"] = error
        results.update(extract_html("", url))
    else:
        results.update(extract_html(html_content, url))
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL to analyze")
    args = parser.parse_args()
    
    try:
        data = analyze_url(args.url)
        print(json.dumps(data))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
