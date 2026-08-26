import pandas as pd
import os
import time
import requests
import urllib.parse
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from worker.extractors.whois_extractor import extract_whois
from worker.extractors.html_extractor import extract_html
from worker.extractors.tls_extractor import extract_tls
from worker.extractors.dns_extractor import extract_dns

def process_url(idx_url):
    idx, url = idx_url
    domain = urllib.parse.urlparse(url).netloc
    if ':' in domain:
        domain = domain.split(':')[0]
        
    try:
        whois_feat = extract_whois(domain)
    except:
        whois_feat = {"domain_age_days": -1, "whois_privacy": 0}
        
    try:
        tls_feat = extract_tls(domain)
    except:
        tls_feat = {"cert_valid": 0, "cert_hostname_match": 0}
        
    try:
        dns_feat = extract_dns(domain)
    except:
        dns_feat = {"has_mx": 0, "has_spf": 0}
        
    dom_rec = {
        "sample_id": idx,
        "url": url,
        "domain": domain,
        "domain_age_days": whois_feat.get("domain_age_days", -1),
        "whois_privacy": whois_feat.get("whois_privacy", 0),
        "cert_valid": tls_feat.get("cert_valid", 0),
        "cert_hostname_match": tls_feat.get("cert_hostname_match", 0),
        "has_mx": dns_feat.get("has_mx", 0),
        "has_spf": dns_feat.get("has_spf", 0),
        "collection_timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        resp = requests.get(url, timeout=1.0, allow_redirects=True, headers={'User-Agent': 'Mozilla/5.0'})
        html_text = resp.text[:50000]
        h_feat = extract_html(html_text, url)
    except Exception:
        h_feat = {
            "password_input_count": 0,
            "external_link_ratio": 0.0,
            "has_redirect": 0,
            "suspicious_text_count": 0,
            "cross_domain_form": 0
        }
        
    html_rec = {
        "sample_id": idx,
        "url": url,
        "password_input_count": h_feat.get("password_input_count", 0),
        "external_link_ratio": h_feat.get("external_link_ratio", 0.0),
        "has_redirect": h_feat.get("has_redirect", 0),
        "suspicious_text_count": h_feat.get("suspicious_text_count", 0),
        "cross_domain_form": h_feat.get("cross_domain_form", 0),
        "collection_timestamp": datetime.utcnow().isoformat()
    }
    
    return dom_rec, html_rec

def main():
    print("Loading test dataset for fixture collection...")
    test_df = pd.read_csv("ml/data/splits/cross_dataset/test.csv")
    
    test_df = pd.concat([
        test_df[test_df['label'] == 0].sample(250, random_state=42),
        test_df[test_df['label'] == 1].sample(80, random_state=42)
    ])
    test_df = test_df.sample(frac=1.0, random_state=42)
    
    urls = test_df['url'].tolist()
    
    domain_records = []
    html_records = []
    
    print(f"Collecting deep evidence for {len(urls)} URLs with ThreadPoolExecutor...")
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_url, (idx, url)): url for idx, url in enumerate(urls)}
        for future in tqdm(as_completed(futures), total=len(urls)):
            try:
                dom_rec, html_rec = future.result()
                domain_records.append(dom_rec)
                html_records.append(html_rec)
            except Exception:
                pass
                
    os.makedirs("experiments/data", exist_ok=True)
    pd.DataFrame(domain_records).to_csv("experiments/data/domain_features.csv", index=False)
    pd.DataFrame(html_records).to_csv("experiments/data/html_features.csv", index=False)
    
    print("Done. Saved real evidence fixtures to experiments/data/")

if __name__ == "__main__":
    main()
