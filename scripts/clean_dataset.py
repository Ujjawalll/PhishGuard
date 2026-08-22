import pandas as pd
import tldextract
import urllib.parse
import os
import json

DATA_DIR = "ml/data/raw"
OUTPUT_DIR = "ml/data/processed"

def normalize_url(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
        # Lowercase scheme and host
        scheme = parsed.scheme.lower() if parsed.scheme else "http"
        netloc = parsed.netloc.lower()
        
        # Decode percent-encoding
        path = urllib.parse.unquote(parsed.path)
        query = urllib.parse.unquote(parsed.query)
        
        # Reconstruct, stripping fragments
        return urllib.parse.urlunparse((scheme, netloc, path, '', query, ''))
    except Exception:
        return url.lower()

def extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ext.domain or ""

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading raw datasets...")
    dfs = []
    for file in ["tranco.csv", "openphish.csv"]:
        path = os.path.join(DATA_DIR, file)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
            
    if not dfs:
        print("No raw datasets found.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"Initial size: {len(df)}")
    
    print("Normalizing URLs...")
    df['normalized_url'] = df['url'].apply(normalize_url)
    
    print("Extracting domains...")
    df['domain'] = df['normalized_url'].apply(extract_domain)
    
    # Check for empty domains
    df = df[df['domain'] != ""]
    
    # 1. Remove exact URL duplicates
    before_len = len(df)
    df = df.drop_duplicates(subset=['normalized_url'])
    print(f"Removed {before_len - len(df)} exact URL duplicates.")
    
    # 2. Detect and resolve label conflicts
    # If the same URL has different labels, we drop it entirely for safety
    conflict_urls = df.groupby('normalized_url')['label'].nunique()
    conflict_urls = conflict_urls[conflict_urls > 1].index
    if len(conflict_urls) > 0:
        print(f"Found {len(conflict_urls)} URLs with conflicting labels. Dropping them.")
        df = df[~df['normalized_url'].isin(conflict_urls)]
        
    # 3. Detect domain-level overlap (Domain in both phishing and legitimate)
    legit_domains = set(df[df['label'] == 0]['domain'])
    phish_domains = set(df[df['label'] == 1]['domain'])
    overlap = legit_domains.intersection(phish_domains)
    print(f"Found {len(overlap)} domains present in both classes.")
    
    # If a domain is in both, it's safer to remove the legitimate entries that share a domain with a phishing entry (e.g. compromised site)
    if overlap:
        drop_mask = (df['label'] == 0) & (df['domain'].isin(overlap))
        df = df[~drop_mask]
        print(f"Dropped {drop_mask.sum()} legitimate URLs from overlapping domains.")
        
    output_path = os.path.join(OUTPUT_DIR, "cleaned_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}. Final size: {len(df)}")
    
    # Save manifest
    manifest = {
        "final_size": len(df),
        "legitimate_count": int((df['label'] == 0).sum()),
        "phishing_count": int((df['label'] == 1).sum()),
        "domain_overlap_removed": len(overlap),
    }
    with open(os.path.join(OUTPUT_DIR, "clean_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    main()
