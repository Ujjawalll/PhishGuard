import pandas as pd
import os
import time
import json
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
from ml.models.registry import get_model_artifact_path
from ml.features.schema import FEATURE_SCHEMA, CURRENT_FEATURE_SCHEMA_VERSION

FEAT_DIR = "ml/data/features/random"

def evaluate_model(model, X_val, y_val):
    start_time = time.time()
    y_pred = model.predict(X_val)
    y_scores = model.predict_proba(X_val)[:, 1]
    inference_time = (time.time() - start_time) / len(X_val)
    
    return {
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "accuracy": accuracy_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_scores),
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
        "inference_time_ms_per_sample": inference_time * 1000
    }

def train_and_evaluate(name, model_estimator, X_train, y_train, X_val, y_val):
    print(f"\nTraining {name}...")
    
    scaler = StandardScaler()
    
    # Use cv=5 for calibration directly, no manual split needed
    if name == "Logistic Regression":
        # LR doesn't strictly need calibration
        calibrated = model_estimator
    else:
        calibrated = CalibratedClassifierCV(estimator=model_estimator, cv=5, method='isotonic')
        
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', calibrated)
    ])
    
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    metrics = evaluate_model(pipeline, X_val, y_val)
    metrics["training_time_s"] = train_time
    
    print(f"{name} Results:")
    print(f"  F1: {metrics['f1']:.4f} | AUC: {metrics['roc_auc']:.4f}")
    
    artifact_dir = get_model_artifact_path(name.lower().replace(" ", "_"), "1.0")
    os.makedirs(artifact_dir, exist_ok=True)
    
    joblib.dump(pipeline, os.path.join(artifact_dir, "model.joblib"))
    
    metadata = {
        "model_name": name,
        "features": FEATURE_SCHEMA,
        "feature_schema_version": CURRENT_FEATURE_SCHEMA_VERSION,
        "metrics": metrics,
        "seed": 42
    }
    with open(os.path.join(artifact_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    return metrics

def main():
    print("Loading extracted features...")
    train_df = pd.read_csv(os.path.join(FEAT_DIR, "train_features.csv"))
    val_df = pd.read_csv(os.path.join(FEAT_DIR, "val_features.csv"))
    
    X_train = train_df[FEATURE_SCHEMA]
    y_train = train_df['label']
    
    X_val = val_df[FEATURE_SCHEMA]
    y_val = val_df['label']
    
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    
    rf = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
    xg = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos)
    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    
    models = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "XGBoost": xg
    }
    
    all_metrics = {}
    for name, estimator in models.items():
        all_metrics[name] = train_and_evaluate(name, estimator, X_train, y_train, X_val, y_val)
        
    os.makedirs("experiments/reports", exist_ok=True)
    report_path = "experiments/reports/model_comparison.csv"
    
    report_df = pd.DataFrame(all_metrics).T
    report_df['TN'] = report_df['confusion_matrix'].apply(lambda x: x[0][0])
    report_df['FP'] = report_df['confusion_matrix'].apply(lambda x: x[0][1])
    report_df['FN'] = report_df['confusion_matrix'].apply(lambda x: x[1][0])
    report_df['TP'] = report_df['confusion_matrix'].apply(lambda x: x[1][1])
    report_df = report_df.drop(columns=['confusion_matrix'])
    
    report_df.to_csv(report_path)
    print(f"\nModel comparison report saved to {report_path}")

if __name__ == "__main__":
    main()
