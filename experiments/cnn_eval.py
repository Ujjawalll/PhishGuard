"""Character-level CNN training & evaluation for PhishGuard.

Deep-learning counterpart to ``experiments/rf_vs_xgb_eval.py`` on the
``feat/phiusiil-model-eval-redesign`` branch. It follows the same job
contract so the redesigned evaluation hooks pick it up unchanged:

* CLI: ``--dataset`` / ``--out`` / ``--quick`` — launched by
  ``backend/app/api/eval.py`` via subprocess with ``PYTHONPATH=<repo root>``.
* Outputs written into ``--out``:
    - ``results.json``      — dataset summary, chosen hyperparameters, and
      ``test_metrics`` keyed by display model name ("Char-CNN")
      with the exact metric keys the redesigned dashboard consumes
      (Accuracy, Precision, Recall, F1-Score, Specificity, Sensitivity,
      ROC-AUC, MCC, PR-AUC, Train-Accuracy, Train-F1), plus per-epoch
      ``training_history`` and an ``error_analysis`` block with false
      positive / false negative counts and examples.
    - ``metrics_matrix.csv`` — metrics table.
    - ``confusion_matrices.png``, ``roc_curves.png``, ``pr_curves.png``,
      ``feature_importances.png`` — the canonical plot names whitelisted
      by the eval API (``PLOT_FILES``), plus ``training_history.png`` and
      ``error_analysis.csv``.
* Model artifacts are persisted to
  ``ml/models/<name>_v1.0_<timestamp>/{model.pt, metadata.json}`` following
  the branch's artifact-directory convention.

Requires PyTorch (``uv add torch`` or ``pip install torch``); every other
dependency is already declared in ``pyproject.toml``.

Example:
    PYTHONPATH=. python experiments/cnn_eval.py \
        --dataset ml/data/processed/PhiUSIIL_Phishing_URL_Dataset.csv \
        --out experiments/reports/char_cnn --quick
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch import Tensor, nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    sys.exit(
        "PyTorch is required for deep-learning training.\n"
        "Install it with:  uv add torch\n"
        "              or:  pip install torch\n"
        "(CPU-only:  pip install torch --index-url https://download.pytorch.org/whl/cpu)"
    )

REPORT_METRICS = [
    "Accuracy", "Precision", "Recall", "F1-Score", "Specificity",
    "Sensitivity", "ROC-AUC", "MCC", "PR-AUC",
]

# Display names used as keys in results.json["test_metrics"]; the redesigned
# ComparisonWorkspace derives its model list from these keys automatically.
MODEL_DISPLAY = {"cnn": "Char-CNN"}
MODEL_ARTIFACT = {"cnn": "char_cnn"}


# --------------------------------------------------------------------------- config
@dataclass(slots=True)
class TrainingConfig:
    """All knobs for one CNN training job (CLI-overridable).

    Attributes mirror the argparse flags; ``quick`` adjusts the defaults
    exactly like ``--quick`` does in ``experiments/rf_vs_xgb_eval.py``.
    """

    model: str = "cnn"                       # {"cnn"}
    dataset: str = os.path.join("ml", "data", "processed", "cleaned_dataset.csv")
    out: str = os.path.join("experiments", "reports", "char_cnn")
    quick: bool = False
    epochs: int = 0                          # 0 -> pick from quick mode
    batch_size: int = 0                      # 0 -> pick from quick mode
    learning_rate: float = 1e-3
    max_url_len: int = 200
    patience: int = 3
    seed: int = 42
    device: str = "auto"                     # {"auto", "cpu", "cuda"}
    max_train_samples: int = 0               # 0 -> pick from quick mode
    threshold: float = 0.5
    save_artifacts: bool = True

    # architecture hyperparameters (kept fixed across runs; recorded in
    # results.json["best_params_*"] and in each artifact's metadata.json)
    embedding_dim: int = 64
    cnn_num_filters: int = 128
    cnn_kernel_sizes: tuple[int, ...] = (3, 5, 7)
    dropout: float = 0.3

    def resolve_defaults(self) -> None:
        """Fill quick-mode-dependent defaults (epochs, batch, train cap)."""
        if self.epochs <= 0:
            self.epochs = 4 if self.quick else 12
        if self.batch_size <= 0:
            self.batch_size = 256 if self.quick else 128
        if self.max_train_samples <= 0:
            self.max_train_samples = 20_000 if self.quick else 100_000

    def arch_params(self, kind: str = "cnn") -> dict[str, Any]:
        """Hyperparameter record for architecture, for results/artifacts."""
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_url_len": self.max_url_len,
            "embedding_dim": self.embedding_dim,
            "dropout": self.dropout,
            "early_stopping_patience": self.patience,
            "seed": self.seed,
            "threshold": self.threshold,
            "num_filters": self.cnn_num_filters,
            "kernel_sizes": list(self.cnn_kernel_sizes),
        }

    def device_name(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- vocabulary
PAD_ID = 0
UNK_ID = 1


@dataclass(slots=True)
class CharVocabulary:
    """Fixed, learned-nothing vocabulary over printable ASCII.

    A fixed vocabulary (no fitting on training text) means encoding is
    deterministic and cannot leak labels, mirroring the guarantee the
    redesign gives ``extract_lexical_features``.
    """

    max_url_len: int
    chars: list[str] = field(default_factory=lambda: [chr(c) for c in range(32, 127)])
    _stoi: dict[str, int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._stoi = {ch: i + 2 for i, ch in enumerate(self.chars)}

    @property
    def size(self) -> int:
        return len(self.chars) + 2  # + PAD + UNK

    def encode(self, url: str) -> np.ndarray:
        """Lowercase, truncate, and map to fixed-length int16 ids."""
        ids = [self._stoi.get(ch, UNK_ID) for ch in url.lower()[: self.max_url_len]]
        arr = np.full(self.max_url_len, PAD_ID, dtype=np.int16)
        arr[: len(ids)] = ids
        return arr

    def encode_batch(self, urls: Sequence[str]) -> np.ndarray:
        return np.stack([self.encode(u) for u in urls]) if urls else np.empty((0, self.max_url_len), np.int16)

    def token_id(self, ch: str) -> int:
        """Id for one character (UNK for anything outside printable ASCII)."""
        return self._stoi.get(ch, UNK_ID)

    @staticmethod
    def char_class(ch: str) -> str:
        """Coarse character class used for the feature-importance plot."""
        if ch.isalpha():
            return "letters"
        if ch.isdigit():
            return "digits"
        if ch in "./-_?=&%;+~:@!$*(),'":
            return "separators"
        return "specials"


# --------------------------------------------------------------------------- data
class URLDataset(Dataset):
    """Torch dataset over pre-encoded fixed-length character matrices."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return (
            torch.from_numpy(self.x[idx].astype(np.int64)),
            torch.tensor(float(self.y[idx])),
        )


def load_urls_and_labels(path: str) -> pd.DataFrame:
    """Load a PhishGuard dataset CSV (url[,label|source]) and normalize it.

    Follows the validation semantics of ``backend/app/api/eval.py``:
    'label' wins, otherwise 'label' is derived from 'source'
    (anything other than 'tranco' is phishing).
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    url_col = "url" if "url" in df.columns else "URL" if "URL" in df.columns else None
    if url_col is None:
        raise SystemExit(f"Dataset must contain a 'url' (or 'URL') column: {path}")
    df = df.rename(columns={url_col: "url"})

    if "label" not in df.columns:
        if "source" not in df.columns:
            raise SystemExit("Dataset must contain a 'label' or a 'source' column")
        df["label"] = (df["source"].str.lower() != "tranco").astype(int)

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)
    bad = set(df["label"].unique()) - {0, 1}
    if bad:
        raise SystemExit(f"label must be binary (0/1), found values: {sorted(bad)}")
    if df["label"].nunique() < 2:
        raise SystemExit("Dataset must contain both classes (label 0 and label 1)")
    if len(df) < 50:
        raise SystemExit(f"Dataset too small: {len(df)} rows (need >= 50)")
    return df[["url", "label"]].drop_duplicates(subset="url").reset_index(drop=True)


def build_splits(
    df: pd.DataFrame, seed: int
) -> tuple[pd.Series, pd.Series, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    """80:10:10 stratified train/val/test split (val drives early stopping)."""
    urls, labels = df["url"], df["label"].to_numpy()
    train_urls, rest_urls, y_train, y_rest = train_test_split(
        urls, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    val_urls, test_urls, y_val, y_test = train_test_split(
        rest_urls, y_rest, test_size=0.5, stratify=y_rest, random_state=seed
    )
    return train_urls, val_urls, test_urls, y_train, y_val, y_test


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, device: str) -> DataLoader:
    """DataLoader with Windows-safe settings (num_workers=0) for job subprocesses."""
    return DataLoader(
        URLDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device == "cuda",
    )


# --------------------------------------------------------------------------- models
class CharCNN(nn.Module):
    """Character-level 1D CNN (URL classifier).

    Parallel Conv1d banks with kernel sizes 3/5/7 act as learned n-gram
    detectors over the character sequence; global max-pooling keeps the
    strongest activation per filter, giving location invariance.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        num_filters: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim, num_filters, k, padding=k // 2) for k in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes), 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens: (B, T) int64 -> logits: (B,) float."""
        x = self.dropout(self.embedding(tokens)).transpose(1, 2)  # (B, E, T)
        feats = [torch.relu(conv(x)).max(dim=2).values for conv in self.convs]
        h = torch.cat(feats, dim=1)
        return self.head(self.dropout(h)).squeeze(-1)


def build_model(kind: str, config: TrainingConfig, vocab_size: int) -> nn.Module:
    if kind == "cnn":
        return CharCNN(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            num_filters=config.cnn_num_filters,
            kernel_sizes=config.cnn_kernel_sizes,
            dropout=config.dropout,
        )
    raise ValueError(f"unknown model kind: {kind}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- training
def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch so runs are reproducible on the same device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def collect_probs(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a loader -> (y_true, p_phishing)."""
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        ys.append(yb.numpy())
        ps.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def compute_branch_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    """Metric block with the exact keys the redesigned dashboard reads."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "Sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "ROC-AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        "MCC": _mcc(tn, fp, fn, tp),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "Train-Accuracy": float("nan"),
        "Train-F1": float("nan"),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def _mcc(tn: int, fp: int, fn: int, tp: int) -> float:
    num = tp * tn - fp * fn
    den = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return num / den if den else 0.0


def train_one_model(
    kind: str,
    config: TrainingConfig,
    vocab: CharVocabulary,
    splits: dict[str, Any],
    device: str,
) -> dict[str, Any] | None:
    """Train architecture end-to-end; return its result block, or None on failure.

    Early stopping restores the best validation-F1 weights; the untouched
    test split is only read after training finishes.
    """
    name = MODEL_DISPLAY[kind]
    print()
    print("=" * 70)
    print(f"4. TRAINING — {name} ({device})")
    print("=" * 70)

    train_urls: pd.Series = splits["train_urls"]
    y_train: np.ndarray = splits["y_train"]
    if len(train_urls) > config.max_train_samples:
        print(f"Subsampling training set to {config.max_train_samples} rows for speed "
              f"(test set stays untouched)...")
        train_urls, _, y_train, _ = train_test_split(
            train_urls, y_train, train_size=config.max_train_samples,
            stratify=y_train, random_state=config.seed,
        )

    x_train = vocab.encode_batch(train_urls.tolist())
    x_val = vocab.encode_batch(splits["val_urls"].tolist())
    x_test = vocab.encode_batch(splits["test_urls"].tolist())

    train_loader = make_loader(x_train, y_train, config.batch_size, True, device)
    val_loader = make_loader(x_val, splits["y_val"], config.batch_size * 2, False, device)
    test_loader = make_loader(x_test, splits["y_test"], config.batch_size * 2, False, device)
    full_train_loader = make_loader(x_train, y_train, config.batch_size * 2, False, device)

    seed_everything(config.seed)
    model = build_model(kind, config, vocab.size).to(device)
    print(f"Parameters: {count_params(model):,}")

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    best_f1, best_state, history, epochs_no_gain = -1.0, None, [], 0
    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        y_val, p_val = collect_probs(model, val_loader, device)
        val_f1 = f1_score(y_val, (p_val >= config.threshold).astype(int), zero_division=0)
        val_auc = roc_auc_score(y_val, p_val) if len(np.unique(y_val)) > 1 else 0.0
        train_loss = float(total_loss) / max(n_batches, 1)
        scheduler.step(val_f1)

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_f1": round(float(val_f1), 4),
            "val_roc_auc": round(float(val_auc), 4),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": round(time.time() - t0, 1),
        }
        history.append(entry)
        print(f"  epoch {entry['epoch']:>2}/{config.epochs}  "
              f"loss {entry['train_loss']:.4f}  val_f1 {entry['val_f1']:.4f}  "
              f"val_auc {entry['val_roc_auc']:.4f}  lr {entry['lr']:.2e}  "
              f"({entry['seconds']}s)")

        if val_f1 > best_f1:
            best_f1, epochs_no_gain = float(val_f1), 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_gain += 1
            if epochs_no_gain >= config.patience:
                print(f"  early stopping after {epoch} epochs (no val-F1 gain for "
                      f"{config.patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_test, p_test = collect_probs(model, test_loader, device)
    y_tr, p_tr = collect_probs(model, full_train_loader, device)
    metrics = compute_branch_metrics(y_test, p_test, config.threshold)
    metrics["Train-Accuracy"] = accuracy_score(y_tr, (p_tr >= config.threshold).astype(int))
    metrics["Train-F1"] = f1_score(y_tr, (p_tr >= config.threshold).astype(int), zero_division=0)

    print(f"\n--- {name}: classification report (TEST SET) ---")
    print(classification_report(y_test, (p_test >= config.threshold).astype(int),
                                target_names=["legitimate", "phishing"], digits=4))
    print(f"--- {name}: confusion matrix (TEST SET) ---")
    print(f"TN={metrics['TN']}  FP={metrics['FP']}\nFN={metrics['FN']}  TP={metrics['TP']}")
    print(f"Overfit check (train vs test F1): {metrics['Train-F1']:.4f} vs "
          f"{metrics['F1-Score']:.4f} (gap {metrics['Train-F1'] - metrics['F1-Score']:+.4f})")

    artifact_dir = save_artifact(kind, model, config, vocab, metrics, best_f1) if config.save_artifacts else None
    return {
        "name": name,
        "kind": kind,
        "model": model,
        "metrics": metrics,
        "history": history,
        "best_val_f1": best_f1,
        "params": config.arch_params(kind),
        "artifact_dir": artifact_dir,
        "y_test": y_test,
        "p_test": p_test,
        "test_urls": splits["test_urls"].tolist(),
    }


# --------------------------------------------------------------------------- artifacts
def save_artifact(
    kind: str,
    model: nn.Module,
    config: TrainingConfig,
    vocab: CharVocabulary,
    metrics: dict[str, Any],
    best_val_f1: float,
) -> str:
    """Persist model.pt + metadata.json under ml/models/ (branch convention)."""
    version = "1.0"
    artifact_dir = os.path.join(
        "ml", "models",
        f"{MODEL_ARTIFACT[kind]}_v{version}_{datetime.now().strftime('%Y%m%dT%H%M%S')}",
    )
    os.makedirs(artifact_dir, exist_ok=True)

    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "arch": config.arch_params(kind),
            "vocab": {"chars": vocab.chars, "max_url_len": vocab.max_url_len,
                      "pad_id": PAD_ID, "unk_id": UNK_ID},
            "model_kind": kind,
        },
        os.path.join(artifact_dir, "model.pt"),
    )
    with open(os.path.join(artifact_dir, "metadata.json"), "w") as f:
        json.dump(
            {
                "model_name": MODEL_DISPLAY[kind],
                "model_version": version,
                "model_type": f"pytorch_{MODEL_ARTIFACT[kind]}",
                "feature_schema_version": "char-v1.0",
                "features": ["char_sequence"],
                "max_url_len": config.max_url_len,
                "vocab_size": vocab.size,
                "hyperparameters": config.arch_params(kind),
                "best_val_f1": best_val_f1,
                "metrics": {k: v for k, v in metrics.items() if not isinstance(v, (np.integer,))},
                "seed": config.seed,
                "training_timestamp": datetime.now().isoformat(),
            },
            f, indent=2, default=str,
        )
    print(f"Artifact saved to {artifact_dir}")
    return artifact_dir


def load_artifact(artifact_dir: str) -> tuple[nn.Module, TrainingConfig, CharVocabulary]:
    """Load a saved artifact for inference (serving hook)."""
    try:
        bundle = torch.load(os.path.join(artifact_dir, "model.pt"), map_location="cpu", weights_only=True)
    except TypeError:  # older torch without weights_only
        bundle = torch.load(os.path.join(artifact_dir, "model.pt"), map_location="cpu")
    arch = bundle["arch"]
    config = TrainingConfig(
        model=bundle["model_kind"],
        epochs=arch["epochs"], batch_size=arch["batch_size"], learning_rate=arch["learning_rate"],
        max_url_len=arch["max_url_len"], patience=arch["early_stopping_patience"], seed=arch["seed"],
        threshold=arch.get("threshold", 0.5), save_artifacts=False,
        embedding_dim=arch["embedding_dim"], dropout=arch["dropout"],
        cnn_num_filters=arch.get("num_filters", 128),
        cnn_kernel_sizes=tuple(arch.get("kernel_sizes", (3, 5, 7))),
    )
    vocab = CharVocabulary(max_url_len=bundle["vocab"]["max_url_len"], chars=bundle["vocab"]["chars"])
    model = build_model(bundle["model_kind"], config, vocab.size)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, config, vocab


@torch.no_grad()
def predict_urls(model: nn.Module, vocab: CharVocabulary, urls: Sequence[str], batch_size: int = 512) -> np.ndarray:
    """P(phishing) for raw URL strings — the serving-side inference hook."""
    probs: list[np.ndarray] = []
    for i in range(0, len(urls), batch_size):
        x = torch.from_numpy(vocab.encode_batch(list(urls[i:i + batch_size])).astype(np.int64))
        probs.append(torch.sigmoid(model(x)).numpy())
    return np.concatenate(probs) if probs else np.empty(0)


# --------------------------------------------------------------------------- plots & analysis
def plot_confusion_matrices(results: list[dict[str, Any]], out_dir: str) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 4.5), squeeze=False)
    for ax, r in zip(axes[0], results):
        m = r["metrics"]
        sns.heatmap(np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]]), annot=True, fmt="d",
                    cmap="Blues", xticklabels=["Pred Legit", "Pred Phish"],
                    yticklabels=["True Legit", "True Phish"], ax=ax, cbar=False)
        ax.set_title(f"{r['name']} — Confusion Matrix (Test)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrices.png"), dpi=150)
    plt.close(fig)


def plot_roc_pr(results: list[dict[str, Any]], out_dir: str) -> None:
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        fpr, tpr, _ = roc_curve(r["y_test"], r["p_test"])
        ax.plot(fpr, tpr, label=f"{r['name']} (AUC={r['metrics']['ROC-AUC']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    base = float(np.mean(results[0]["y_test"]))
    for r in results:
        prec, rec, _ = precision_recall_curve(r["y_test"], r["p_test"])
        ax.plot(rec, prec, label=f"{r['name']} (AP={r['metrics']['PR-AUC']:.4f})")
    ax.axhline(base, color="k", ls="--", lw=0.8, label=f"Baseline ({base:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Test Set"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "pr_curves.png"), dpi=150); plt.close(fig)


def plot_feature_importances(results: list[dict[str, Any]], vocab: CharVocabulary, out_dir: str) -> None:
    """Character-class signal strength from learned embeddings.

    Mean L2 norm of embedding rows grouped by character class is a cheap,
    faithful proxy for which input groups the network leans on — the DNN
    analog of tree feature importances, under the canonical filename the
    eval API whitelists.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 4.5), squeeze=False)
    for ax, r in zip(axes[0], results):
        norms = None
        for module in r["model"].modules():
            if isinstance(module, nn.Embedding):
                norms = module.weight.detach().norm(dim=1).numpy()
                break
        classes = ["letters", "digits", "separators", "specials"]
        means = {
            cls: float(np.mean([norms[vocab.token_id(c)] for c in vocab.chars if CharVocabulary.char_class(c) == cls]))
            for cls in classes
        }
        pd.Series(means).plot.barh(ax=ax, color="#2878b5")
        ax.set_title(f"{r['name']} — Character-Class Signal (embedding L2 norm)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_importances.png"), dpi=150)
    plt.close(fig)


def plot_training_history(results: list[dict[str, Any]], out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in results:
        epochs = [h["epoch"] for h in r["history"]]
        axes[0].plot(epochs, [h["train_loss"] for h in r["history"]], marker=".", label=r["name"])
        axes[1].plot(epochs, [h["val_f1"] for h in r["history"]], marker=".", label=r["name"])
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss"); axes[0].legend()
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation F1"); axes[1].legend()
    fig.suptitle("Training History")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_history.png"), dpi=150)
    plt.close(fig)


def error_analysis(r: dict[str, Any]) -> dict[str, Any]:
    """Track false positives / false negatives (redesigned eval hook)."""
    y_pred = (r["p_test"] >= 0.5).astype(int)
    urls = np.array(r["test_urls"], dtype=object)
    fp_idx = np.where((r["y_test"] == 0) & (y_pred == 1))[0]
    fn_idx = np.where((r["y_test"] == 1) & (y_pred == 0))[0]

    def examples(idx: np.ndarray) -> list[dict[str, Any]]:
        return [
            {"url": str(urls[i])[:120], "true": int(r["y_test"][i]), "pred": int(y_pred[i]), "prob": round(float(r["p_test"][i]), 4)}
            for i in idx[:25]
        ]

    return {
        "false_positives": int(len(fp_idx)),
        "false_negatives": int(len(fn_idx)),
        "examples_false_positives": examples(fp_idx),
        "examples_false_negatives": examples(fn_idx),
    }


# --------------------------------------------------------------------------- main
def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=os.path.join("ml", "data", "processed", "cleaned_dataset.csv"))
    parser.add_argument("--out", default=os.path.join("experiments", "reports", "char_cnn"))
    parser.add_argument("--quick", action="store_true", help="fewer epochs + subsampled training for faster runs")
    parser.add_argument("--model", choices=["cnn"], default="cnn")
    parser.add_argument("--epochs", type=int, default=0, help="0 = quick/full default")
    parser.add_argument("--batch-size", type=int, default=0, help="0 = quick/full default")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-url-len", type=int, default=200)
    parser.add_argument("--patience", type=int, default=3, help="early-stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-train-samples", type=int, default=0, help="cap on training rows (0 = default)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-save-artifacts", action="store_true")
    a = parser.parse_args()
    config = TrainingConfig(
        model=a.model, dataset=a.dataset, out=a.out, quick=a.quick,
        epochs=a.epochs, batch_size=a.batch_size, learning_rate=a.learning_rate,
        max_url_len=a.max_url_len, patience=a.patience, seed=a.seed, device=a.device,
        max_train_samples=a.max_train_samples, threshold=a.threshold,
        save_artifacts=not a.no_save_artifacts,
    )
    config.resolve_defaults()
    return config


def main() -> int:
    config = parse_args()
    os.makedirs(config.out, exist_ok=True)
    device = config.device_name()

    print("=" * 70)
    print("1. DATASET SUMMARY")
    print("=" * 70)
    df = load_urls_and_labels(config.dataset)
    n_pos, n_neg = int((df.label == 1).sum()), int((df.label == 0).sum())
    print(f"Rows (deduped): {len(df)} | Columns: {list(df.columns)}")
    print(f"Class imbalance ratio (neg:pos): {n_neg / n_pos:.1f}:1 ({100 * n_pos / len(df):.2f}% phishing)")
    truncated = int((df['url'].str.len() > config.max_url_len).sum())
    print(f"URLs longer than max_url_len={config.max_url_len} (truncated): {truncated}")

    print()
    print("=" * 70)
    print("2. PREPROCESSING — CHARACTER ENCODING & SPLITS")
    print("=" * 70)
    seed_everything(config.seed)
    train_urls, val_urls, test_urls, y_train, y_val, y_test = build_splits(df, config.seed)
    print(f"Train: {len(train_urls)} ({int(y_train.sum())} pos) | "
          f"Val: {len(val_urls)} ({int(y_val.sum())} pos) | "
          f"Test: {len(test_urls)} ({int(y_test.sum())} pos)  [80:10:10, stratified, seed={config.seed}]")
    vocab = CharVocabulary(config.max_url_len)
    print(f"Vocabulary: fixed printable ASCII, size={vocab.size} (PAD={PAD_ID}, UNK={UNK_ID})")

    print()
    print("=" * 70)
    print("3. MODEL ARCHITECTURE")
    print("=" * 70)
    print(f"Char-CNN : embedding({config.embedding_dim}) -> parallel Conv1d "
          f"({'/'.join(map(str, config.cnn_kernel_sizes))} x {config.cnn_num_filters} filters) "
          f"-> global max-pool -> MLP")
    print(f"Device: {device} | epochs={config.epochs} (early stop patience={config.patience}) | "
          f"batch={config.batch_size} | lr={config.learning_rate}")

    splits = {
        "train_urls": train_urls, "val_urls": val_urls, "test_urls": test_urls,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }
    kinds = ["cnn"]
    results: list[dict[str, Any]] = []
    for kind in kinds:
        try:
            r = train_one_model(kind, config, vocab, splits, device)
            if r is not None:
                results.append(r)
        except Exception as exc:
            print(f"\n[ERROR] training {MODEL_DISPLAY[kind]} failed: {exc}")
            traceback.print_exc()
    if not results:
        print("No model trained successfully — job failed.")
        return 1

    print()
    print("=" * 70)
    print("5. EVALUATION ON UNTOUCHED TEST SET")
    print("=" * 70)
    header = f"{'Metric':<14}" + "".join(f"{r['name']:>12}" for r in results)
    print(header)
    for k in REPORT_METRICS:
        print(f"{k:<14}" + "".join(f"{r['metrics'][k]:>12.4f}" for r in results))

    print()
    print("=" * 70)
    print("6. ERROR ANALYSIS — FALSE POSITIVES / FALSE NEGATIVES")
    print("=" * 70)
    analysis: dict[str, Any] = {}
    err_rows: list[dict[str, Any]] = []
    for r in results:
        analysis[r["name"]] = error_analysis(r)
        a = analysis[r["name"]]
        print(f"{r['name']}: FP={a['false_positives']}  FN={a['false_negatives']}  "
              f"(top examples saved to error_analysis.csv)")
        y_pred = (r["p_test"] >= config.threshold).astype(int)
        for i in np.where(y_pred != r["y_test"])[0][:25]:
            err_rows.append({
                "model": r["name"], "url": str(r["test_urls"][i])[:200],
                "true_label": int(r["y_test"][i]), "pred_label": int(y_pred[i]),
                "prob_phishing": round(float(r["p_test"][i]), 4),
                "error_type": "FP" if r["y_test"][i] == 0 else "FN",
            })
    if err_rows:
        pd.DataFrame(err_rows).to_csv(os.path.join(config.out, "error_analysis.csv"), index=False)

    print()
    print("=" * 70)
    print("7. PLOTS & ARTIFACTS")
    print("=" * 70)
    plot_confusion_matrices(results, config.out)
    plot_roc_pr(results, config.out)
    plot_feature_importances(results, vocab, config.out)
    plot_training_history(results, config.out)

    payload: dict[str, Any] = {
        "dataset": {
            "file": config.dataset,
            "rows": len(df), "positive": n_pos, "negative": n_neg,
            "features": ["char_sequence"],
            "train_size": len(train_urls), "val_size": len(val_urls), "test_size": len(test_urls),
            "max_url_len": config.max_url_len, "vocab_size": vocab.size,
            "truncated_urls": truncated,
        },
        "test_metrics": {r["name"]: r["metrics"] for r in results},
        "training_history": {r["name"]: r["history"] for r in results},
        "error_analysis": analysis,
        "device": device,
        "artifacts": {r["name"]: r["artifact_dir"] for r in results if r["artifact_dir"]},
    }
    for r in results:
        payload[f"best_params_{r['kind']}"] = r["params"]
        payload[f"best_val_f1_{r['kind']}"] = r["best_val_f1"]
    with open(os.path.join(config.out, "results.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)
    pd.DataFrame(
        {r["name"]: {m: round(r["metrics"][m], 4) for m in REPORT_METRICS + ["Train-F1", "Train-Accuracy"]}
         for r in results}
    ).to_csv(os.path.join(config.out, "metrics_matrix.csv"))

    print(f"\nSaved: {config.out}/{{results.json, metrics_matrix.csv, error_analysis.csv, "
          f"confusion_matrices.png, roc_curves.png, pr_curves.png, feature_importances.png, training_history.png}}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
