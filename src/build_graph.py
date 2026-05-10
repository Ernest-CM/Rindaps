"""
Stage 2 — Build the global PyPI dependency graph from cached JSON files.

Design note (different from plan.md §4):
  Plan.md proposed (package, version) nodes. PyPI's main JSON only carries
  requires_dist for the LATEST version of each package. Producing per-version
  edges would require ~10-30x more HTTP requests (one per version per package).
  We instead use package-level nodes whose features summarize the package's
  full version history, and we store the full version catalog separately
  (data/graph/catalog.json) so Stage 3 can sample concrete (package, version)
  configurations and Stage 4 can inject configuration-specific version info
  as runtime features when extracting subgraphs.

Outputs:
  data/graph/pypi_graph.pt   — torch_geometric.data.Data (x, edge_index, edge_attr)
  data/graph/node_index.json — {package_name: node_id}
  data/graph/catalog.json    — {package_name: {versions, latest, ...}}
  data/graph/summary.json    — graph-level statistics

Usage:
  python -m src.build_graph
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version
from torch_geometric.data import Data

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUT_DIR = Path("data/graph")


@dataclass
class PackageMeta:
    name: str
    versions: list[str] = field(default_factory=list)  # all parseable, non-prerelease
    latest: str | None = None
    latest_major: int = 0
    first_release: datetime | None = None
    latest_release: datetime | None = None
    requires_dist: list[str] = field(default_factory=list)
    num_classifiers: int = 0
    has_python_requires: bool = False


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_package_json(path: Path) -> PackageMeta | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    info = doc.get("info") or {}
    raw_name = (info.get("name") or path.stem).lower()
    meta = PackageMeta(name=raw_name)

    # Iterate all releases, collect parseable non-prerelease versions and earliest upload time per version
    releases = doc.get("releases") or {}
    parsed: list[tuple[Version, str, datetime | None]] = []
    for ver_str, files in releases.items():
        try:
            v = Version(ver_str)
        except InvalidVersion:
            continue
        if v.is_prerelease or v.is_devrelease:
            continue
        upload_times = [
            _parse_iso(f.get("upload_time_iso_8601") or f.get("upload_time"))
            for f in (files or [])
        ]
        upload_times = [t for t in upload_times if t is not None]
        earliest = min(upload_times) if upload_times else None
        parsed.append((v, ver_str, earliest))

    if not parsed:
        return None

    parsed.sort(key=lambda x: x[0])
    meta.versions = [p[1] for p in parsed]
    meta.latest = parsed[-1][1]
    meta.latest_major = parsed[-1][0].major

    upload_times = [p[2] for p in parsed if p[2] is not None]
    if upload_times:
        meta.first_release = min(upload_times)
        meta.latest_release = max(upload_times)

    meta.requires_dist = list(info.get("requires_dist") or [])
    meta.num_classifiers = len(info.get("classifiers") or [])
    meta.has_python_requires = bool(info.get("requires_python"))
    return meta


def _is_runtime_dep(req: Requirement) -> bool:
    """Skip dependencies that only apply for extras (e.g. test/docs).

    A bare runtime dependency has either no marker or a marker that does NOT
    require an extra. requires_dist with `extra == "test"` is install-time
    optional and would not be present in a default install, so we ignore it.
    """
    if req.marker is None:
        return True
    return "extra" not in str(req.marker)


def _build_edges(
    catalog: dict[str, PackageMeta],
    node_id: dict[str, int],
) -> tuple[list[tuple[int, int]], list[str]]:
    edges: list[tuple[int, int]] = []
    edge_specs: list[str] = []
    skipped_unknown = 0
    skipped_unparseable = 0
    skipped_extras = 0
    for name, meta in catalog.items():
        src = node_id[name]
        for req_str in meta.requires_dist:
            try:
                req = Requirement(req_str)
            except InvalidRequirement:
                skipped_unparseable += 1
                continue
            dep = req.name.lower()
            if dep not in node_id:
                skipped_unknown += 1
                continue
            if not _is_runtime_dep(req):
                skipped_extras += 1
                continue
            edges.append((src, node_id[dep]))
            edge_specs.append(str(req.specifier))
    logging.info(
        "edges: kept=%d skipped_unknown=%d skipped_extras=%d skipped_unparseable=%d",
        len(edges), skipped_unknown, skipped_extras, skipped_unparseable,
    )
    return edges, edge_specs


def _build_features(
    catalog: dict[str, PackageMeta],
    node_id: dict[str, int],
    in_deg: np.ndarray,
    out_deg: np.ndarray,
) -> np.ndarray:
    now = datetime.now(timezone.utc)
    feats = np.zeros((len(node_id), 8), dtype=np.float32)
    for name, meta in catalog.items():
        i = node_id[name]
        first_age = (now - meta.first_release).days if meta.first_release else 0
        latest_age = (now - meta.latest_release).days if meta.latest_release else 0
        feats[i] = [
            float(len(meta.versions)),
            float(first_age),
            float(latest_age),
            float(in_deg[i]),
            float(out_deg[i]),
            1.0 if meta.has_python_requires else 0.0,
            float(meta.latest_major),
            float(meta.num_classifiers),
        ]
    return feats


def _normalize(features: np.ndarray) -> np.ndarray:
    """Log-scale heavy-tailed columns and z-score the rest."""
    out = features.copy()
    # Heavy-tailed: num_versions(0), first_age(1), latest_age(2), in_deg(3), out_deg(4), classifiers(7)
    for col in (0, 1, 2, 3, 4, 7):
        out[:, col] = np.log1p(out[:, col])
    # latest_major (col 6): clip to [0, 50] for stability
    out[:, 6] = np.clip(out[:, 6], 0, 50)
    # z-score everything except the binary col 5
    for col in range(out.shape[1]):
        if col == 5:
            continue
        m, s = out[:, col].mean(), out[:, col].std()
        if s > 1e-9:
            out[:, col] = (out[:, col] - m) / s
    return out.astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build PyPI dependency graph.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    _setup_logging()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    json_files = [p for p in args.raw_dir.glob("*.json") if p.name != "top-pypi-packages.json"]
    logging.info("parsing %d cached JSON files", len(json_files))

    catalog: dict[str, PackageMeta] = {}
    skipped = 0
    for i, path in enumerate(json_files, 1):
        meta = _parse_package_json(path)
        if meta is None:
            skipped += 1
            continue
        catalog[meta.name] = meta
        if i % 1000 == 0:
            logging.info("  parsed %d/%d", i, len(json_files))
    logging.info("catalog size: %d (skipped %d)", len(catalog), skipped)

    nodes = sorted(catalog.keys())
    node_id = {n: i for i, n in enumerate(nodes)}

    edges, edge_specs = _build_edges(catalog, node_id)

    in_deg = np.zeros(len(nodes), dtype=np.float64)
    out_deg = np.zeros(len(nodes), dtype=np.float64)
    for s, d in edges:
        out_deg[s] += 1
        in_deg[d] += 1

    raw_feats = _build_features(catalog, node_id, in_deg, out_deg)
    feats = _normalize(raw_feats)
    logging.info("feature matrix: shape=%s", feats.shape)

    if edges:
        edge_index = torch.tensor(
            [[s for s, _ in edges], [d for _, d in edges]],
            dtype=torch.long,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    x = torch.tensor(feats, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    graph_path = args.out_dir / "pypi_graph.pt"
    torch.save(data, graph_path)
    logging.info("wrote %s", graph_path)

    (args.out_dir / "node_index.json").write_text(
        json.dumps(node_id, indent=2), encoding="utf-8"
    )

    catalog_serializable = {
        name: {
            "versions": meta.versions,
            "latest": meta.latest,
            "latest_major": meta.latest_major,
            "first_release": meta.first_release.isoformat() if meta.first_release else None,
            "latest_release": meta.latest_release.isoformat() if meta.latest_release else None,
            "num_classifiers": meta.num_classifiers,
            "has_python_requires": meta.has_python_requires,
        }
        for name, meta in catalog.items()
    }
    (args.out_dir / "catalog.json").write_text(
        json.dumps(catalog_serializable, indent=2), encoding="utf-8"
    )

    edge_specs_path = args.out_dir / "edge_specifiers.json"
    edge_specs_path.write_text(json.dumps(edge_specs), encoding="utf-8")

    summary = {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "feature_dim": int(feats.shape[1]),
        "mean_in_degree": float(in_deg.mean()),
        "mean_out_degree": float(out_deg.mean()),
        "max_in_degree": int(in_deg.max() if len(in_deg) else 0),
        "max_out_degree": int(out_deg.max() if len(out_deg) else 0),
        "isolated_nodes": int(((in_deg == 0) & (out_deg == 0)).sum()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("summary: %s", summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
