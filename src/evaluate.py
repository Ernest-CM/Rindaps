"""
Stage 5 — Evaluation: load all four trained models, compute test-set metrics
on the seed-42 split, and produce the figures used in Chapter 4.

Outputs:
  reports/metrics.json
  reports/figures/roc_curves.png
  reports/figures/pr_curves.png
  reports/figures/confusion_matrices.png
  reports/figures/recall_by_category.png
  reports/figures/f1_by_size.png

Usage:
  python -m src.evaluate
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import logging
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.loader import DataLoader

from src.features import load_graph, materialize_dataset
from src.models.gat import GATClassifier
from src.models.gcn import GCNClassifier

warnings.filterwarnings("ignore", message=".*torch-scatter.*")

LABELED_PATH = Path("data/labeled/labeled.csv")
SPLIT_PATH = Path("data/splits/split_42.json")
CHECKPOINT_DIR = Path("checkpoints")
REPORTS_DIR = Path("reports")
FIG_DIR = REPORTS_DIR / "figures"

MODEL_DISPLAY = {
    "rf": "Random Forest",
    "svm": "SVM (RBF)",
    "gcn": "GCN",
    "gat": "GAT",
}
MODEL_COLORS = {
    "rf": "#1f77b4",
    "svm": "#ff7f0e",
    "gcn": "#2ca02c",
    "gat": "#d62728",
}
MODEL_ORDER = ["rf", "svm", "gcn", "gat"]


def _setup() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")


def _load_split() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    return (np.array(split["train"]), np.array(split["val"]), np.array(split["test"]))


def _predict_baseline(name: str, X: np.ndarray) -> np.ndarray:
    with (CHECKPOINT_DIR / f"{name}_best.pkl").open("rb") as f:
        bundle = pickle.load(f)
    return bundle["model"].predict_proba(X)[:, 1]


@torch.no_grad()
def _predict_gnn(name: str, subgraphs: list, device: torch.device) -> np.ndarray:
    ckpt = torch.load(CHECKPOINT_DIR / f"{name}_best.pt",
                      map_location=device, weights_only=False)
    cls = GCNClassifier if name == "gcn" else GATClassifier
    kwargs = dict(
        num_nodes=ckpt["num_nodes"],
        in_channels=ckpt["in_channels"],
        hidden_channels=ckpt["hidden"],
        embed_dim=ckpt["embed_dim"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    )
    if name == "gat":
        kwargs["heads"] = ckpt["heads"]
    model = cls(**kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    loader = DataLoader(subgraphs, batch_size=128, shuffle=False)
    probs: list[float] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch, batch.node_id)
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.array(probs, dtype=np.float32)


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.025):
        pred = (y_prob >= thr).astype(np.int64)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def _plot_roc(probs_test: dict[str, np.ndarray], y_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in MODEL_ORDER:
        fpr, tpr, _ = roc_curve(y_test, probs_test[name])
        auc = roc_auc_score(y_test, probs_test[name])
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2,
                label=f"{MODEL_DISPLAY[name]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the held-out test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curves.png", dpi=200)
    plt.close(fig)


def _plot_pr(probs_test: dict[str, np.ndarray], y_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    base_rate = float(y_test.mean())
    for name in MODEL_ORDER:
        prec, rec, _ = precision_recall_curve(y_test, probs_test[name])
        ap = average_precision_score(y_test, probs_test[name])
        ax.plot(rec, prec, color=MODEL_COLORS[name], lw=2,
                label=f"{MODEL_DISPLAY[name]} (AP={ap:.3f})")
    ax.axhline(base_rate, ls="--", color="gray", lw=1,
               label=f"Random ({base_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curves on the test set")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr_curves.png", dpi=200)
    plt.close(fig)


def _plot_confusion(probs_test: dict, thresholds: dict, y_test: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, name in zip(axes.flat, MODEL_ORDER):
        thr = thresholds[name]
        pred = (probs_test[name] >= thr).astype(np.int64)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Compatible", "Conflict"],
                    yticklabels=["Compatible", "Conflict"])
        ax.set_title(f"{MODEL_DISPLAY[name]} (thr={thr:.2f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion matrices on test set (val-tuned thresholds)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_recall_by_category(
    probs_test: dict, thresholds: dict, y_test: np.ndarray, meta_test: list[dict],
) -> None:
    categories: dict[str, list[int]] = {}
    for i, m in enumerate(meta_test):
        if y_test[i] != 1:
            continue
        cat = m["failure_category"] or "uncategorized"
        categories.setdefault(cat, []).append(i)
    cats = sorted(categories, key=lambda c: -len(categories[c]))
    cats = [c for c in cats if len(categories[c]) >= 5]  # drop tiny categories

    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.2
    x = np.arange(len(cats))
    for j, name in enumerate(MODEL_ORDER):
        thr = thresholds[name]
        pred = (probs_test[name] >= thr).astype(np.int64)
        recalls = [float(pred[categories[c]].mean()) for c in cats]
        ax.bar(x + (j - 1.5) * width, recalls, width=width,
               color=MODEL_COLORS[name], label=MODEL_DISPLAY[name])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={len(categories[c])})" for c in cats], fontsize=9)
    ax.set_ylabel("Recall (catch rate of true conflicts)")
    ax.set_title("Per-category recall on test set")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "recall_by_category.png", dpi=200)
    plt.close(fig)


def _plot_f1_by_size(
    probs_test: dict, thresholds: dict, y_test: np.ndarray, meta_test: list[dict],
) -> None:
    sizes_present = sorted({m["n_pkgs"] for m in meta_test})
    by_size_idx = {s: [i for i, m in enumerate(meta_test) if m["n_pkgs"] == s]
                   for s in sizes_present}

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name in MODEL_ORDER:
        thr = thresholds[name]
        pred = (probs_test[name] >= thr).astype(np.int64)
        f1s = []
        for s in sizes_present:
            idx = by_size_idx[s]
            yt, yp = y_test[idx], pred[idx]
            f1s.append(f1_score(yt, yp, zero_division=0))
        ax.plot(sizes_present, f1s, marker="o",
                color=MODEL_COLORS[name], label=MODEL_DISPLAY[name])
    ax.set_xlabel("Configuration size (number of packages)")
    ax.set_ylabel("F1")
    ax.set_title("F1 by configuration size on test set (tuned thresholds)")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f1_by_size.png", dpi=200)
    plt.close(fig)


def main() -> int:
    _setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("device: %s", device)

    logging.info("loading PyPI graph + dataset...")
    bundle = load_graph()
    subgraphs, hc_features, labels, meta = materialize_dataset(
        LABELED_PATH, bundle, k=2,
        exclude_categories=["missing_distribution", "invalid_specifier"],
    )
    _, val_idx, test_idx = _load_split()
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    meta_test = [meta[i] for i in test_idx]
    logging.info("test set: %d configs  pos=%d neg=%d",
                 len(y_test), int(y_test.sum()), int((y_test == 0).sum()))

    probs_val: dict[str, np.ndarray] = {}
    probs_test: dict[str, np.ndarray] = {}
    for name in ("rf", "svm"):
        probs_val[name] = _predict_baseline(name, hc_features[val_idx])
        probs_test[name] = _predict_baseline(name, hc_features[test_idx])
    val_subs = [subgraphs[i] for i in val_idx]
    test_subs = [subgraphs[i] for i in test_idx]
    for name in ("gcn", "gat"):
        probs_val[name] = _predict_gnn(name, val_subs, device)
        probs_test[name] = _predict_gnn(name, test_subs, device)

    thresholds = {n: _best_threshold(y_val, probs_val[n]) for n in MODEL_ORDER}

    metrics: dict[str, dict] = {}
    for name in MODEL_ORDER:
        thr = thresholds[name]
        prob = probs_test[name]
        pred_default = (prob >= 0.5).astype(np.int64)
        pred_tuned = (prob >= thr).astype(np.int64)
        metrics[name] = {
            "tuned_threshold": float(thr),
            "test_at_default_threshold": _eval_metrics(y_test, pred_default, prob),
            "test_at_tuned_threshold": _eval_metrics(y_test, pred_tuned, prob),
        }
        m_tuned = metrics[name]["test_at_tuned_threshold"]
        logging.info(
            "%-4s  AUC=%.3f  PR-AUC=%.3f  F1=%.3f  prec=%.3f  recall=%.3f  thr=%.2f",
            name, m_tuned["roc_auc"], m_tuned["pr_auc"], m_tuned["f1"],
            m_tuned["precision"], m_tuned["recall"], thr,
        )

    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps({
            "test_set_size": int(len(y_test)),
            "test_positives": int(y_test.sum()),
            "test_negatives": int((y_test == 0).sum()),
            "thresholds": thresholds,
            "models": metrics,
        }, indent=2),
        encoding="utf-8",
    )
    logging.info("wrote reports/metrics.json")

    logging.info("rendering figures...")
    _plot_roc(probs_test, y_test)
    _plot_pr(probs_test, y_test)
    _plot_confusion(probs_test, thresholds, y_test)
    _plot_recall_by_category(probs_test, thresholds, y_test, meta_test)
    _plot_f1_by_size(probs_test, thresholds, y_test, meta_test)
    logging.info("wrote figures to %s", FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
