import pandas as pd
import tldextract
import urllib.parse
import os
import json
import random

DATA_DIR = "ml/data/raw"
OUTPUT_DIR = "ml/data/processed"

def normalize_url(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
        scheme = parsed.scheme.lower() if parsed.scheme else "http"
        netloc = parsed.netloc.lower()
        path = urllib.parse.unquote(parsed.path)
        query = urllib.parse.unquote(parsed.query)
        return urllib.parse.urlunparse((scheme, netloc, path, '', query, ''))
    except Exception:
        return url.lower()

def extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ext.domain or ""

def augment_legit_url(url: str) -> str:
    """Randomly append paths, queries, and subdomains to prevent ML structural bias"""
    if random.random() < 0.2:
        return url # 20% remain naked
        
    # 50% chance to add www.
    if random.random() < 0.5:
        url = url.replace("http://", "http://www.").replace("https://", "https://www.")
        
    paths = [
        "/", "/index.html", "/about", "/contact", "/login", "/register", 
        "/search", "/products/view/123", "/api/v1/users", "/blog/article-name-here",
        "/wp-content/uploads/image.jpg", "/assets/css/style.css", "/download"
    ]
    queries = [
        "", "?q=test", "?id=123", "?ref=twitter", "?page=2", "?lang=en",
        "?token=abcxyz123", "?utm_source=newsletter&utm_medium=email",
        "?redirect_url=https%3A%2F%2Fexample.com%2Fdashboard&session_id=abcdef1234567890abcdef1234567890",
        "?action=verify&token=abcxyz123abcxyz123abcxyz123abcxyz123abcxyz123abcxyz123abcxyz123abcxyz123"
    ]
    
    path = random.choice(paths)
    query = random.choice(queries) if random.random() < 0.5 else ""
    
    if url.endswith("/"):
        url = url[:-1]
        
    return url + path + query

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)
    
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
    
    # Augment legit URLs to prevent shortcut learning
    print("Augmenting legitimate URLs to prevent structural bias...")
    mask = df['label'] == 0
    df.loc[mask, 'url'] = df.loc[mask, 'url'].apply(augment_legit_url)
    
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
    
    if overlap:
        drop_mask = (df['label'] == 0) & (df['domain'].isin(overlap))
        df = df[~drop_mask]
        print(f"Dropped {drop_mask.sum()} legitimate URLs from overlapping domains.")
        
    output_path = os.path.join(OUTPUT_DIR, "cleaned_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}. Final size: {len(df)}")
    
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
