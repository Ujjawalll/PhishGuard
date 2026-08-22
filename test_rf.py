import os
from ml.features.lexical import extract_lexical_features
import joblib
from glob import glob
import pandas as pd
import json

def test_url(url, pipeline, feature_cols):
    features = extract_lexical_features(url)
    df = pd.DataFrame([features])[feature_cols]
    ml_prob = pipeline.predict_proba(df)[0][1]
    print(f"{url:40s} | Prob: {ml_prob:.4f}")

paths = glob("ml/models/random_forest_*")
latest_path = sorted(paths)[-1]
pipeline = joblib.load(os.path.join(latest_path, "model.joblib"))

with open(os.path.join(latest_path, "metadata.json")) as f:
    feature_cols = json.load(f)["features"]

urls = [
    "https://www.google.com/search?q=test",
    "https://apple.com/iphone-16",
    "http://secure-paypal-login.com.info/login.php",
    "https://github.com/login",
    "http://192.168.1.1/admin"
]

for u in urls:
    test_url(u, pipeline, feature_cols)
