"""
Stage 4 — Training entrypoint.

Trains one of: gcn, gat, rf, svm.
GNN models train on subgraphs around each configuration.
RF/SVM baselines train on hand-crafted features (12-dim).

Outputs:
  checkpoints/{model}_best.pt          — best GNN model state_dict (or sklearn pickle)
  checkpoints/{model}_history.json     — per-epoch metrics for GNN runs
  data/splits/split_{seed}.json        — reproducible train/val/test indices

Usage:
  python -m src.train --model gcn  --epochs 80 --batch-size 64
  python -m src.train --model gat  --epochs 80 --batch-size 64
  python -m src.train --model rf
  python -m src.train --model svm
"""
from __future__ import annotations

import os
# Set BEFORE importing torch — handles the Intel OpenMP duplicate on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from src.features import load_graph, materialize_dataset
from src.models.baselines import make_random_forest, make_svm
from src.models.gat import GATClassifier
from src.models.gcn import GCNClassifier

LABELED_PATH = Path("data/labeled/labeled.csv")
SPLIT_DIR = Path("data/splits")
CHECKPOINT_DIR = Path("checkpoints")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _stratified_split(
    n: int,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n)
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, stratify=labels, random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=labels[temp_idx], random_state=seed,
    )
    return train_idx, val_idx, test_idx


def _save_split(train_idx, val_idx, test_idx, seed: int) -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLIT_DIR / f"split_{seed}.json"
    path.write_text(json.dumps({
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }), encoding="utf-8")


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
    }


def _train_gnn(
    model_name: str,
    subgraphs: list,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    num_nodes: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("device: %s", device)

    train_set = [subgraphs[i] for i in train_idx]
    val_set = [subgraphs[i] for i in val_idx]
    test_set = [subgraphs[i] for i in test_idx]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    in_channels = subgraphs[0].x.size(1)
    if model_name == "gcn":
        model = GCNClassifier(num_nodes=num_nodes, in_channels=in_channels,
                              hidden_channels=args.hidden, embed_dim=args.embed_dim,
                              num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "gat":
        model = GATClassifier(num_nodes=num_nodes, in_channels=in_channels,
                              hidden_channels=args.hidden, embed_dim=args.embed_dim,
                              heads=args.heads, num_layers=args.num_layers, dropout=args.dropout)
    else:
        raise ValueError(f"unknown GNN model: {model_name}")
    model = model.to(device)

    n_pos = float((labels[train_idx] == 1).sum())
    n_neg = float((labels[train_idx] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    logging.info("class balance — train pos=%d neg=%d  pos_weight=%.3f", int(n_pos), int(n_neg), pos_weight.item())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict] = []
    best_val_f1 = -1.0
    best_state: dict | None = None
    patience_left = args.patience
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch, batch.node_id)
            target = batch.y.view(-1).float()
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * target.size(0)
            n_seen += target.size(0)
        train_loss = total_loss / max(n_seen, 1)

        val_metrics = _evaluate_gnn(model, val_loader, device)
        logging.info(
            "epoch %3d  loss=%.4f  val_acc=%.3f  val_f1=%.3f  val_auc=%.3f",
            epoch, train_loss, val_metrics["accuracy"], val_metrics["f1"], val_metrics["roc_auc"],
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logging.info("early stopping at epoch %d (best val_f1=%.3f)", epoch, best_val_f1)
                break

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    val_true, val_prob = _gnn_probs(model, val_loader, device)
    best_thr, best_thr_f1 = _best_threshold(val_true, val_prob)
    logging.info("threshold tuning on val: best F1=%.3f at threshold=%.3f",
                 best_thr_f1, best_thr)

    val_metrics_tuned = _evaluate_gnn(model, val_loader, device, threshold=best_thr)
    test_metrics = _evaluate_gnn(model, test_loader, device, threshold=best_thr)
    test_metrics_default = _evaluate_gnn(model, test_loader, device, threshold=0.5)
    logging.info("val metrics  @ thr=%.3f: %s", best_thr, val_metrics_tuned)
    logging.info("test metrics @ thr=%.3f: %s", best_thr, test_metrics)
    logging.info("test metrics @ thr=0.500: %s", test_metrics_default)

    ckpt_path = CHECKPOINT_DIR / f"{model_name}_best.pt"
    torch.save({
        "state_dict": best_state,
        "num_nodes": num_nodes,
        "in_channels": in_channels,
        "embed_dim": args.embed_dim,
        "hidden": args.hidden,
        "num_layers": args.num_layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "model_name": model_name,
    }, ckpt_path)
    logging.info("wrote %s", ckpt_path)

    (CHECKPOINT_DIR / f"{model_name}_history.json").write_text(
        json.dumps({
            "history": history,
            "test_at_tuned_threshold": test_metrics,
            "test_at_0.5_threshold": test_metrics_default,
            "tuned_threshold": best_thr,
            "best_val_f1": best_val_f1,
        }, indent=2),
        encoding="utf-8",
    )


@torch.no_grad()
def _gnn_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch, batch.node_id)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y_prob.extend(prob.tolist())
        y_true.extend(batch.y.view(-1).cpu().numpy().tolist())
    return np.array(y_true, dtype=np.int64), np.array(y_prob, dtype=np.float32)


def _evaluate_gnn(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                  threshold: float = 0.5) -> dict:
    y_true, y_prob = _gnn_probs(model, loader, device)
    y_pred = (y_prob >= threshold).astype(np.int64)
    return _eval_metrics(y_true, y_pred, y_prob)


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds 0.05..0.95 and return (best_threshold, best_F1)."""
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.025):
        pred = (y_prob >= thr).astype(np.int64)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


def _train_baseline(
    model_name: str,
    hc_features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> None:
    if model_name == "rf":
        model = make_random_forest(seed=seed)
    elif model_name == "svm":
        model = make_svm(seed=seed)
    else:
        raise ValueError(f"unknown baseline: {model_name}")

    X_train = hc_features[train_idx]
    y_train = labels[train_idx]
    X_val = hc_features[val_idx]
    y_val = labels[val_idx]
    X_test = hc_features[test_idx]
    y_test = labels[test_idx]

    logging.info("fitting %s on %d samples", model_name, len(X_train))
    t0 = time.monotonic()
    model.fit(X_train, y_train)
    logging.info("  fit done in %.1fs", time.monotonic() - t0)

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    best_thr, best_thr_f1 = _best_threshold(y_val, val_prob)
    logging.info("threshold tuning on val: best F1=%.3f at threshold=%.3f",
                 best_thr_f1, best_thr)

    for split_name, y, prob in [("val", y_val, val_prob), ("test", y_test, test_prob)]:
        pred_default = (prob >= 0.5).astype(np.int64)
        pred_tuned = (prob >= best_thr).astype(np.int64)
        logging.info("%s metrics (%s) @ thr=0.500: %s",
                     model_name, split_name, _eval_metrics(y, pred_default, prob))
        logging.info("%s metrics (%s) @ thr=%.3f: %s",
                     model_name, split_name, best_thr,
                     _eval_metrics(y, pred_tuned, prob))

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{model_name}_best.pkl"
    with ckpt_path.open("wb") as f:
        pickle.dump({"model": model, "feature_names": [
            "n_pkgs", "n_pinned", "frac_known", "sub_n_nodes", "sub_n_edges",
            "sub_avg_deg", "node_feat0", "node_feat1", "node_feat2",
            "node_feat3", "node_feat4", "node_feat5",
            "pin_total", "pin_known", "pin_unknown",
            "pin_mean_pos", "pin_min_pos", "pin_n_old",
        ]}, f)
    logging.info("wrote %s", ckpt_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train install-failure prediction models.")
    parser.add_argument("--model", choices=["gcn", "gat", "rf", "svm"], required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-hop", type=int, default=2)
    parser.add_argument(
        "--label-filter",
        choices=["all", "graph"],
        default="graph",
        help="'graph' (default) drops missing_distribution and invalid_specifier "
             "failures (which depend on wheel availability, not graph structure); "
             "'all' keeps every labeled configuration.",
    )
    args = parser.parse_args(argv)

    _setup_logging()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.info("loading PyPI graph...")
    bundle = load_graph()
    logging.info("graph: %d nodes, %d edges, %d-dim features",
                 bundle.data.num_nodes, bundle.data.edge_index.size(1), bundle.data.x.size(1))

    logging.info("materializing dataset from %s (label-filter=%s) ...",
                 LABELED_PATH, args.label_filter)
    exclude = (["missing_distribution", "invalid_specifier"]
               if args.label_filter == "graph" else None)
    subgraphs, hc_features, labels, _meta = materialize_dataset(
        LABELED_PATH, bundle, k=args.k_hop, exclude_categories=exclude,
    )
    logging.info("dataset: %d configs  pos=%d neg=%d",
                 len(subgraphs), int((labels == 1).sum()), int((labels == 0).sum()))

    train_idx, val_idx, test_idx = _stratified_split(len(subgraphs), labels, seed=args.seed)
    _save_split(train_idx, val_idx, test_idx, seed=args.seed)
    logging.info("split: train=%d  val=%d  test=%d", len(train_idx), len(val_idx), len(test_idx))

    if args.model in ("gcn", "gat"):
        _train_gnn(
            args.model, subgraphs, labels, train_idx, val_idx, test_idx, args,
            num_nodes=bundle.data.num_nodes,
        )
    else:
        _train_baseline(args.model, hc_features, labels, train_idx, val_idx, test_idx, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
