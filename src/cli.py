"""
Stage 6 — `conflict-predict` CLI.

Given a requirements.txt path, prints the GAT-predicted probability that
the configuration would fail to install due to a graph-structural
dependency conflict, plus a HIGH/MEDIUM/LOW risk band.

Usage:
  python -m src.cli path/to/requirements.txt
  python -m src.cli path/to/requirements.txt --model gcn --threshold 0.32
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path

import torch
from packaging.requirements import InvalidRequirement, Requirement

from src.features import bare_name, build_subgraph, load_graph
from src.models.gat import GATClassifier
from src.models.gcn import GCNClassifier

CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_MODEL = "gat"
DEFAULT_THRESHOLD_PATH = Path("reports/metrics.json")


def _read_requirements(path: Path) -> list[str]:
    pkgs: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        try:
            req = Requirement(line)
        except InvalidRequirement:
            pkgs.append(line)
            continue
        spec = str(req.specifier)
        pkgs.append(f"{req.name}{spec}" if spec else req.name)
    return pkgs


def _load_model(name: str, device: torch.device):
    ckpt_path = CHECKPOINT_DIR / f"{name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Train first: python -m src.train --model {name}"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    return model.to(device).eval()


def _resolve_threshold(model_name: str, override: float | None) -> float:
    if override is not None:
        return float(override)
    if DEFAULT_THRESHOLD_PATH.exists():
        try:
            doc = json.loads(DEFAULT_THRESHOLD_PATH.read_text(encoding="utf-8"))
            t = doc.get("thresholds", {}).get(model_name)
            if t is not None:
                return float(t)
        except (json.JSONDecodeError, OSError):
            pass
    return 0.5


def _risk_band(prob: float, threshold: float) -> str:
    if prob >= max(threshold + 0.20, 0.70):
        return "HIGH"
    if prob >= threshold:
        return "MEDIUM"
    return "LOW"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict pip-install conflict risk for a requirements file.")
    parser.add_argument("requirements", type=Path, help="path to requirements.txt")
    parser.add_argument("--model", choices=["gcn", "gat"], default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the threshold (defaults to the val-tuned one from reports/metrics.json)")
    args = parser.parse_args(argv)

    if not args.requirements.exists():
        print(f"error: requirements file not found: {args.requirements}", file=sys.stderr)
        return 2

    packages = _read_requirements(args.requirements)
    if not packages:
        print("error: no requirements parsed from file", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_graph()
    subgraph = build_subgraph(packages, bundle, k=2)

    if subgraph is None:
        print("error: none of the listed packages are in the trained model's vocabulary",
              file=sys.stderr)
        return 3

    known: list[str] = []
    unknown: list[str] = []
    for spec in packages:
        name = bare_name(spec)
        (known if name in bundle.node_index else unknown).append(spec)

    model = _load_model(args.model, device)
    threshold = _resolve_threshold(args.model, args.threshold)

    with torch.no_grad():
        sub = subgraph.to(device)
        batch = torch.zeros(sub.num_nodes, dtype=torch.long, device=device)
        logits = model(sub.x, sub.edge_index, batch, sub.node_id)
        prob = float(torch.sigmoid(logits).item())

    band = _risk_band(prob, threshold)
    print(f"Conflict probability: {prob:.3f}  ({band} risk)")
    print(f"Decision threshold:   {threshold:.3f}  (model={args.model})")
    print(f"Verdict:              {'CONFLICT predicted' if prob >= threshold else 'COMPATIBLE predicted'}")
    print()
    print(f"Packages parsed:      {len(packages)}")
    print(f"  known to graph:     {len(known)}")
    if unknown:
        print(f"  not in vocabulary:  {len(unknown)}  ({', '.join(unknown[:5])}{'...' if len(unknown) > 5 else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
