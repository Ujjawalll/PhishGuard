"""RF vs XGBoost rigorous evaluation on PhishGuard cleaned_dataset.csv."""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
)
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

from ml.features.lexical import extract_lexical_features
from ml.features.schema import FEATURE_SCHEMA

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=os.path.join("ml", "data", "processed", "cleaned_dataset.csv"))
parser.add_argument("--out", default=os.path.join("experiments", "reports", "rf_vs_xgb"))
parser.add_argument("--quick", action="store_true", help="fewer search iterations for faster runs")
args = parser.parse_args()

REPORT_DIR = args.out
N_ITER = 10 if args.quick else 25
os.makedirs(REPORT_DIR, exist_ok=True)
RANDOM_STATE = 42

# ---------------------------------------------------------------- load + inspect
print("=" * 70)
print("1. DATASET SUMMARY")
print("=" * 70)
df = pd.read_csv(args.dataset)
print(f"Rows: {len(df)} | Columns: {list(df.columns)}")
print(f"Target column: 'label' -> {dict(df['label'].value_counts())}")
n_pos, n_neg = int((df.label == 1).sum()), int((df.label == 0).sum())
print(f"Class imbalance ratio (neg:pos): {n_neg / n_pos:.1f}:1  "
      f"({100 * n_pos / len(df):.2f}% positive)")
print(f"Missing values per column:\n{df.isna().sum().to_string()}")
print(f"Duplicate rows: {df.duplicated().sum()} | Duplicate URLs: {df['url'].duplicated().sum()}")

# ---------------------------------------------------------------- preprocessing
print()
print("=" * 70)
print("2. PREPROCESSING")
print("=" * 70)
# - 'rank' dropped: only exists for Tranco rows (300 missing) and encodes source,
#   not phishing signal -> would leak the label.
# - 'source' dropped: directly encodes the label (tranco=0, openphish=1).
# - 'normalized_url'/'domain' dropped: string duplicates of 'url', no extra info.
# - Features: 18 lexical features extracted from the URL string (deterministic
#   function, no fitting -> no leakage possible).
# - Duplicates: none found, nothing removed.
# - Missing values in features: filled with 0 after extraction if any.
feat_rows = [extract_lexical_features(u) for u in df["url"]]
X = pd.DataFrame(feat_rows)[FEATURE_SCHEMA]
y = df["label"]
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} lexical features")
print(f"Features: {FEATURE_SCHEMA}")
print(f"Missing feature values after extraction: {int(X.isna().sum().sum())}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {len(X_train)} ({int(y_train.sum())} pos) | "
      f"Test: {len(X_test)} ({int(y_test.sum())} pos)  [80:20, stratified, seed={RANDOM_STATE}]")

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ---------------------------------------------------------------- RF tuning
print()
print("=" * 70)
print("3. RANDOM FOREST — RandomizedSearchCV (5-fold, scoring=F1)")
print("=" * 70)
rf_pipe = Pipeline([("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))])
rf_dist = {
    "clf__n_estimators": randint(100, 501),
    "clf__max_depth": [None, 8, 12, 16, 24],
    "clf__min_samples_split": randint(2, 11),
    "clf__min_samples_leaf": randint(1, 6),
    "clf__max_features": ["sqrt", "log2", None],
    "clf__class_weight": ["balanced", "balanced_subsample", None],
}
rf_search = RandomizedSearchCV(
    rf_pipe, rf_dist, n_iter=N_ITER, scoring="f1", cv=cv,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
)
rf_search.fit(X_train, y_train)
print(f"Best CV F1: {rf_search.best_score_:.4f}")
print(f"Best params: {json.dumps({k.replace('clf__', ''): v for k, v in rf_search.best_params_.items()}, default=str)}")
rf = rf_search.best_estimator_

# ---------------------------------------------------------------- XGB tuning
print()
print("=" * 70)
print("4. XGBOOST — RandomizedSearchCV (5-fold, scoring=F1)")
print("=" * 70)
xgb_pipe = Pipeline([
    ("scaler", StandardScaler()),  # fitted on train folds only inside CV
    ("clf", XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                          eval_metric="logloss", scale_pos_weight=scale_pos)),
])
xgb_dist = {
    "clf__n_estimators": randint(100, 501),
    "clf__max_depth": randint(3, 11),
    "clf__learning_rate": uniform(0.01, 0.29),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__colsample_bytree": uniform(0.6, 0.4),
    "clf__min_child_weight": randint(1, 8),
    "clf__gamma": uniform(0, 2),
    "clf__reg_lambda": uniform(0.1, 3),
}
xgb_search = RandomizedSearchCV(
    xgb_pipe, xgb_dist, n_iter=N_ITER, scoring="f1", cv=cv,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
)
xgb_search.fit(X_train, y_train)
print(f"Best CV F1: {xgb_search.best_score_:.4f}")
print(f"Best params: {json.dumps({k.replace('clf__', ''): (round(v, 4) if isinstance(v, float) else v) for k, v in xgb_search.best_params_.items()}, default=str)}")
xgb = xgb_search.best_estimator_

# ---------------------------------------------------------------- evaluation
def full_eval(model, name, X_tr, y_tr, X_te, y_te):
    y_pred_tr = model.predict(X_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    m = {
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred, zero_division=0),
        "Recall": recall_score(y_te, y_pred),
        "F1-Score": f1_score(y_te, y_pred),
        "Specificity": tn / (tn + fp),
        "Sensitivity": tp / (tp + fn),
        "ROC-AUC": roc_auc_score(y_te, y_prob),
        "MCC": matthews_corrcoef(y_te, y_pred),
        "PR-AUC": average_precision_score(y_te, y_prob),
        "Train-Accuracy": accuracy_score(y_tr, y_pred_tr),
        "Train-F1": f1_score(y_tr, y_pred_tr),
    }
    print(f"\n--- {name}: classification report (TEST SET) ---")
    print(classification_report(y_te, y_pred, target_names=["legitimate", "phishing"], digits=4))
    print(f"--- {name}: confusion matrix (TEST SET) ---")
    print(f"TN={tn}  FP={fp}\nFN={fn}  TP={tp}")
    return m, y_pred, y_prob, (tn, fp, fn, tp)

print()
print("=" * 70)
print(f"5. EVALUATION ON UNTOUCHED TEST SET ({len(X_test)} rows)")
print("=" * 70)
rf_m, rf_pred, rf_prob, rf_cm = full_eval(rf, "Random Forest", X_train, y_train, X_test, y_test)
xgb_m, xgb_pred, xgb_prob, xgb_cm = full_eval(xgb, "XGBoost", X_train, y_train, X_test, y_test)

# ---------------------------------------------------------------- comparison table
print()
print("=" * 70)
print("6. SIDE-BY-SIDE PERFORMANCE MATRIX")
print("=" * 70)
metrics_order = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity",
                 "Sensitivity", "ROC-AUC", "MCC", "PR-AUC"]
print(f"{'Metric':<14}{'Random Forest':>15}{'XGBoost':>15}")
for k in metrics_order:
    print(f"{k:<14}{rf_m[k]:>15.4f}{xgb_m[k]:>15.4f}")
print(f"\nOverfit check (train vs test F1):")
print(f"  Random Forest: train {rf_m['Train-F1']:.4f} vs test {rf_m['F1-Score']:.4f} "
      f"(gap {rf_m['Train-F1'] - rf_m['F1-Score']:+.4f})")
print(f"  XGBoost:       train {xgb_m['Train-F1']:.4f} vs test {xgb_m['F1-Score']:.4f} "
      f"(gap {xgb_m['Train-F1'] - xgb_m['F1-Score']:+.4f})")

# ---------------------------------------------------------------- plots
sns.set_style("whitegrid")

# confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (tn, fp, fn, tp), name in zip(axes, [rf_cm, xgb_cm], ["Random Forest", "XGBoost"]):
    sns.heatmap(np.array([[tn, fp], [fn, tp]]), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Legit", "Pred Phish"], yticklabels=["True Legit", "True Phish"],
                ax=ax, cbar=False)
    ax.set_title(f"{name} — Confusion Matrix (Test)")
fig.tight_layout()
fig.savefig(os.path.join(REPORT_DIR, "confusion_matrices.png"), dpi=150)
plt.close(fig)

# ROC curves
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
fpr_xg, tpr_xg, _ = roc_curve(y_test, xgb_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={rf_m['ROC-AUC']:.4f})")
ax.plot(fpr_xg, tpr_xg, label=f"XGBoost (AUC={xgb_m['ROC-AUC']:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Test Set"); ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(REPORT_DIR, "roc_curves.png"), dpi=150)
plt.close(fig)

# PR curves
prec_rf, rec_rf, _ = precision_recall_curve(y_test, rf_prob)
prec_xg, rec_xg, _ = precision_recall_curve(y_test, xgb_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec_rf, prec_rf, label=f"Random Forest (AP={rf_m['PR-AUC']:.4f})")
ax.plot(rec_xg, prec_xg, label=f"XGBoost (AP={xgb_m['PR-AUC']:.4f})")
ax.axhline(y_test.mean(), color="k", ls="--", lw=0.8, label=f"Baseline ({y_test.mean():.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — Test Set"); ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(REPORT_DIR, "pr_curves.png"), dpi=150)
plt.close(fig)

# feature importances
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
rf_imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=FEATURE_SCHEMA).sort_values()
rf_imp.tail(12).plot.barh(ax=axes[0], color="#2878b5")
axes[0].set_title("Random Forest — Top Feature Importances (Gini)")
xgb_imp = pd.Series(xgb.named_steps["clf"].feature_importances_, index=FEATURE_SCHEMA).sort_values()
xgb_imp.tail(12).plot.barh(ax=axes[1], color="#c82423")
axes[1].set_title("XGBoost — Top Feature Importances (gain)")
fig.tight_layout()
fig.savefig(os.path.join(REPORT_DIR, "feature_importances.png"), dpi=150)
plt.close(fig)

print(f"\nTop-5 features (RF):  {list(rf_imp.tail(5).index[::-1])}")
print(f"Top-5 features (XGB): {list(xgb_imp.tail(5).index[::-1])}")

# ---------------------------------------------------------------- save artifacts
results = {
    "dataset": {
        "file": args.dataset,
        "rows": len(df), "positive": n_pos, "negative": n_neg,
        "features": FEATURE_SCHEMA,
        "train_size": len(X_train), "test_size": len(X_test),
    },
    "best_params_rf": {k.replace("clf__", ""): v for k, v in rf_search.best_params_.items()},
    "best_cv_f1_rf": rf_search.best_score_,
    "best_params_xgb": {k.replace("clf__", ""): v for k, v in xgb_search.best_params_.items()},
    "best_cv_f1_xgb": xgb_search.best_score_,
    "test_metrics": {"Random Forest": rf_m, "XGBoost": xgb_m},
}
with open(os.path.join(REPORT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
pd.DataFrame({k: {m: round(v[m], 4) for m in metrics_order + ["Train-F1", "Train-Accuracy"]}
              for k, v in [("Random Forest", rf_m), ("XGBoost", xgb_m)]}).to_csv(
    os.path.join(REPORT_DIR, "metrics_matrix.csv"))
print(f"\nSaved: {REPORT_DIR}/{{confusion_matrices.png, roc_curves.png, pr_curves.png, feature_importances.png, results.json, metrics_matrix.csv}}")
print("DONE")
