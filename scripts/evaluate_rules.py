import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os
import numpy as np
from ml.rules.engine import RuleEngine
from ml.features.lexical import extract_lexical_features
from tqdm import tqdm

def main():
    val_path = "ml/data/splits/random/val.csv"
    if not os.path.exists(val_path):
        print(f"Validation data not found at {val_path}")
        return

    df = pd.read_csv(val_path)
    engine = RuleEngine()
    
    y_true = df['label'].values
    y_scores = []
    
    print(f"Evaluating Rule Engine on {len(df)} validation samples...")
    for url in tqdm(df['url']):
        features = extract_lexical_features(url)
        res = engine.evaluate(url, features)
        y_scores.append(res["normalized_score"])
        
    y_scores = np.array(y_scores)
    
    # Tune T_RULE to maximize F1 on validation data
    best_f1 = -1
    best_threshold = 0.0
    thresholds = np.linspace(0, 1, 101)
    
    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            
    # Calculate final metrics with best_threshold
    y_pred = (y_scores >= best_threshold).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    print("\n--- Rule Engine Evaluation Baseline ---")
    print(f"Rule Threshold (T_RULE): {best_threshold:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print("Confusion Matrix:")
    print(f"TN: {tn} | FP: {fp}")
    print(f"FN: {fn} | TP: {tp}")
    
    # Save results to experiments folder
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/rule_baseline.txt", "w") as f:
        f.write(f"T_RULE: {best_threshold:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"FPR: {fpr:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")

if __name__ == "__main__":
    main()
