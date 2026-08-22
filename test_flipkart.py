import pandas as pd
from ml.features.lexical import extract_lexical_features
import joblib
from glob import glob
import json
import os

features = extract_lexical_features("https://www.flipkart.com/desidekho-mug-keychain-gift-set/p/itm7ef8342bf73d6")
paths = glob("ml/models/xgboost_*")
latest_path = sorted(paths)[-1]
pipeline = joblib.load(os.path.join(latest_path, "model.joblib"))
with open(os.path.join(latest_path, "metadata.json")) as f:
    feature_cols = json.load(f)["features"]
df = pd.DataFrame([features])[feature_cols]
print("Flipkart Prob:", pipeline.predict_proba(df)[0][1])
