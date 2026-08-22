import httpx
import os
import json
from datetime import datetime
import pandas as pd
import zipfile
import io

DATA_DIR = "ml/data/raw"

def download_tranco():
    print("Downloading Tranco (legitimate)...")
    url = "https://tranco-list.eu/top-1m.csv.zip"
    try:
        response = httpx.get(url, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                # Read just the top 10,000 to save time/memory for dev
                df = pd.read_csv(f, names=["rank", "url"], nrows=10000)
                
        # Add http:// to domains to make them URLs
        df['url'] = "http://" + df['url']
        df['label'] = 0
        df['source'] = "tranco"
        
        output_path = os.path.join(DATA_DIR, "tranco.csv")
        df.to_csv(output_path, index=False)
        
        return {
            "source": "Tranco",
            "url": url,
            "license": "Open Data Commons Open Database License (ODbL)",
            "collection_date": datetime.now().isoformat(),
            "size": len(df),
            "file": "tranco.csv"
        }
    except Exception as e:
        print(f"Failed to download Tranco: {e}")
        return None

def download_openphish():
    print("Downloading OpenPhish (phishing)...")
    url = "https://openphish.com/feed.txt"
    try:
        response = httpx.get(url, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
        
        urls = response.text.strip().split('\n')
        df = pd.DataFrame({"url": urls})
        df['label'] = 1
        df['source'] = "openphish"
        
        output_path = os.path.join(DATA_DIR, "openphish.csv")
        df.to_csv(output_path, index=False)
        
        return {
            "source": "OpenPhish",
            "url": url,
            "license": "Non-commercial use only (Free Feed)",
            "collection_date": datetime.now().isoformat(),
            "size": len(df),
            "file": "openphish.csv"
        }
    except Exception as e:
        print(f"Failed to download OpenPhish: {e}")
        return None

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    manifest = []
    
    tranco_meta = download_tranco()
    if tranco_meta:
        manifest.append(tranco_meta)
        
    op_meta = download_openphish()
    if op_meta:
        manifest.append(op_meta)
        
    with open(os.path.join(DATA_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("Download complete.")

if __name__ == "__main__":
    main()
