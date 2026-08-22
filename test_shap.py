import os
import joblib
import json
from glob import glob
from ml.features.lexical import extract_lexical_features
from ml.explainability.explainer import Explainer

paths = glob("ml/models/xgboost_*")
latest_path = sorted(paths)[-1]
pipeline = joblib.load(os.path.join(latest_path, "model.joblib"))
with open(os.path.join(latest_path, "metadata.json")) as f:
    feature_cols = json.load(f)["features"]

explainer = Explainer(pipeline, feature_cols)
features = extract_lexical_features("https://www.google.com/search?q=test")
shap_vals = explainer.get_shap_values(features)

for feature, score in sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"{feature}: {score:.4f}")
