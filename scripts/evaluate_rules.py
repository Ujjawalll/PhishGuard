import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os
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
    y_pred = []
    y_scores = []
    
    print(f"Evaluating Rule Engine on {len(df)} validation samples...")
    for url in tqdm(df['url']):
        features = extract_lexical_features(url)
        res = engine.evaluate(url, features)
        # Binary prediction: 1 if HIGH_RISK or SUSPICIOUS (i.e. normalized_score > safe threshold)
        y_pred.append(1 if res["risk_level"] in ["HIGH_RISK", "SUSPICIOUS"] else 0)
        y_scores.append(res["normalized_score"])
        
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n--- Rule Engine Evaluation Baseline ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print("Confusion Matrix:")
    print(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")
    
    # Save results to experiments folder
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/rule_baseline.txt", "w") as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")

if __name__ == "__main__":
    main()
