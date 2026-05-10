"""
Stage 4 — Feature extraction and dataset assembly.

Provides:
  - bare_name(spec)           — strip version specifier from "requests==2.31"
  - load_graph()              — load D1 (PyPI dependency graph)
  - build_subgraph(...)       — k-hop subgraph around a configuration's nodes,
                                with a binary "in_config" indicator added as
                                an extra node feature column.
  - hand_crafted_features(...) — fixed-length feature vector per config for
                                 RF/SVM baselines.
  - ConfigDataset             — torch_geometric InMemoryDataset that holds
                                all labeled configurations as Data objects.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from packaging.requirements import InvalidRequirement, Requirement
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


class _ConfigSubgraph(Data):
    """Data subclass that stops PyG from auto-incrementing `node_id` during
    batch collation. Without this override, PyG treats long-tensor attributes
    as graph-local indices and shifts them by cumulative num_nodes when
    batching multiple subgraphs — which scrambles our embedding lookups.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "node_id":
            return 0
        return super().__inc__(key, value, *args, **kwargs)

GRAPH_PATH = Path("data/graph/pypi_graph.pt")
NODE_INDEX_PATH = Path("data/graph/node_index.json")
CATALOG_PATH = Path("data/graph/catalog.json")


def bare_name(spec: str) -> str:
    """'requests==2.31.0' -> 'requests'.  Tolerates malformed input."""
    s = spec.strip()
    try:
        return Requirement(s).name.lower()
    except InvalidRequirement:
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            if sep in s:
                return s.split(sep, 1)[0].strip().lower()
        return s.lower()


def is_pinned(spec: str) -> bool:
    return "==" in spec


@dataclass
class GraphBundle:
    data: Data                       # full PyPI graph
    node_index: dict[str, int]
    catalog: dict[str, dict]


def load_graph() -> GraphBundle:
    data = torch.load(GRAPH_PATH, weights_only=False)
    with NODE_INDEX_PATH.open("r", encoding="utf-8") as f:
        node_index = json.load(f)
    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        catalog = json.load(f)
    return GraphBundle(data=data, node_index=node_index, catalog=catalog)


def config_to_node_ids(packages: list[str], node_index: dict[str, int]) -> list[int]:
    out: list[int] = []
    for spec in packages:
        name = bare_name(spec)
        if name in node_index:
            out.append(node_index[name])
    return out


def build_subgraph(
    packages: list[str],
    bundle: GraphBundle,
    k: int = 2,
) -> Data | None:
    """Return the k-hop subgraph around the configuration's known nodes.

    Each node carries 8 graph-level features (from build_graph) plus 4
    config-aware features that inject the version-pin signal:
        [8]  in_config      — 1.0 if the node is named in the configuration
        [9]  is_pinned      — 1.0 if pinned with `==` in this configuration
        [10] pin_position   — normalized position of the pinned version in
                              the package's release history (0=oldest,
                              1=newest); 0.5 if unpinned/missing
        [11] is_unknown_ver — 1.0 if pinned to a version not in our catalog
                              (strong signal for missing_distribution)

    Returns None if no package in the configuration is known to the graph.
    """
    pin_info: dict[int, tuple[float, float, float, float]] = {}
    for spec in packages:
        name, ver = _split_pin(spec)
        if name not in bundle.node_index:
            continue
        nid = bundle.node_index[name]
        if ver is None:
            pin_info[nid] = (1.0, 0.0, 0.5, 0.0)
            continue
        meta = bundle.catalog.get(name) or {}
        versions = meta.get("versions") or []
        if versions and ver in versions:
            denom = max(len(versions) - 1, 1)
            pin_info[nid] = (1.0, 1.0, versions.index(ver) / denom, 0.0)
        else:
            pin_info[nid] = (1.0, 1.0, 0.0, 1.0)

    if not pin_info:
        return None

    seed_ids = list(pin_info.keys())
    seed = torch.tensor(seed_ids, dtype=torch.long)
    sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
        seed,
        num_hops=k,
        edge_index=bundle.data.edge_index,
        relabel_nodes=True,
        num_nodes=bundle.data.num_nodes,
        flow="source_to_target",
        directed=False,
    )
    sub_x = bundle.data.x[sub_nodes]
    cfg_feats = torch.zeros(sub_x.size(0), 4, dtype=sub_x.dtype)
    for orig_id, sub_idx in zip(seed_ids, mapping.tolist()):
        cfg_feats[sub_idx] = torch.tensor(pin_info[orig_id], dtype=sub_x.dtype)
    sub_x = torch.cat([sub_x, cfg_feats], dim=1)

    return _ConfigSubgraph(
        x=sub_x, edge_index=sub_edge_index, node_id=sub_nodes.long(),
    )


def _split_pin(spec: str) -> tuple[str, str | None]:
    """'requests==2.31.0' -> ('requests', '2.31.0').  Returns (name, None) if unpinned."""
    if "==" in spec:
        a, _, b = spec.partition("==")
        return a.strip().lower(), b.strip()
    return bare_name(spec), None


def version_features(packages: list[str], bundle: GraphBundle) -> np.ndarray:
    """6-dim feature vector capturing how 'extreme' the version pins in this
    configuration are. This is where the install-failure signal lives.

    Dims:
      [0] total pinned packages
      [1] pinned to a version that exists in our catalog
      [2] pinned to a version NOT in our catalog (strong missing_distribution signal)
      [3] mean normalized position of known pins (0=oldest, 1=newest)
      [4] min  normalized position of known pins
      [5] count of pins in the oldest 30% of the package's release history
    """
    positions: list[float] = []
    n_known_pins = 0
    n_unknown_pins = 0
    for spec in packages:
        name, ver = _split_pin(spec)
        if ver is None:
            continue
        meta = bundle.catalog.get(name)
        if meta is None:
            continue
        versions = meta.get("versions") or []
        if not versions:
            continue
        if ver in versions:
            denom = max(len(versions) - 1, 1)
            positions.append(versions.index(ver) / denom)
            n_known_pins += 1
        else:
            n_unknown_pins += 1

    if positions:
        mean_pos = float(np.mean(positions))
        min_pos = float(np.min(positions))
        n_old = float(sum(1 for p in positions if p < 0.30))
    else:
        mean_pos = 0.5
        min_pos = 0.5
        n_old = 0.0

    return np.array(
        [n_known_pins + n_unknown_pins, n_known_pins, n_unknown_pins,
         mean_pos, min_pos, n_old],
        dtype=np.float32,
    )


def hand_crafted_features(
    packages: list[str],
    bundle: GraphBundle,
    k: int = 2,
) -> np.ndarray:
    """Fixed 18-dim feature vector per configuration for RF/SVM baselines."""
    n_pkgs = len(packages)
    n_pinned = sum(1 for p in packages if is_pinned(p))
    n_known = 0
    pkg_node_features: list[np.ndarray] = []
    for spec in packages:
        name = bare_name(spec)
        if name in bundle.node_index:
            n_known += 1
            row = bundle.data.x[bundle.node_index[name]].numpy()
            pkg_node_features.append(row)

    sub = build_subgraph(packages, bundle, k=k)
    if sub is None:
        sub_n_nodes = sub_n_edges = 0
        sub_avg_deg = 0.0
    else:
        sub_n_nodes = int(sub.num_nodes)
        sub_n_edges = int(sub.edge_index.size(1))
        sub_avg_deg = (sub_n_edges / max(sub_n_nodes, 1)) * 2.0

    if pkg_node_features:
        stacked = np.stack(pkg_node_features, axis=0)
        mean_node = stacked.mean(axis=0)
    else:
        mean_node = np.zeros(bundle.data.x.size(1), dtype=np.float32)

    ver_feats = version_features(packages, bundle)

    feats = np.zeros(18, dtype=np.float32)
    # [0..5] structural
    feats[0] = n_pkgs
    feats[1] = n_pinned
    feats[2] = n_known / max(n_pkgs, 1)
    feats[3] = sub_n_nodes
    feats[4] = sub_n_edges
    feats[5] = sub_avg_deg
    # [6..11] aggregated node features over known packages
    feats[6:12] = mean_node[:6]
    # [12..17] version-pin features (the missing piece)
    feats[12:18] = ver_feats
    return feats


def materialize_dataset(
    labeled_csv_path: Path,
    bundle: GraphBundle,
    k: int = 2,
    exclude_categories: list[str] | None = None,
) -> tuple[list[Data], np.ndarray, np.ndarray, list[dict]]:
    """Read labeled.csv and produce parallel arrays:
       - subgraphs:   list[Data] for GNN training
       - hc_features: np.ndarray [N, 18] for baseline training
       - labels:      np.ndarray [N]
       - meta:        list of dicts (config_id, source, failure_category)

    Configurations whose packages are *all* unknown to the graph are dropped.
    If exclude_categories is given, configs whose failure_category matches any
    listed value are also dropped — used to filter out missing_distribution
    failures that depend on wheel availability rather than graph structure.
    """
    import pandas as pd

    df = pd.read_csv(labeled_csv_path)
    if exclude_categories:
        n_before = len(df)
        cat = df["failure_category"].fillna("")
        df = df[~cat.isin(exclude_categories)].reset_index(drop=True)
        print(f"filtered out {n_before - len(df)} configs "
              f"with failure_category in {exclude_categories}")
    subgraphs: list[Data] = []
    feats_rows: list[np.ndarray] = []
    labels: list[int] = []
    meta: list[dict] = []
    dropped = 0

    for _, row in df.iterrows():
        try:
            packages = json.loads(row["packages_json"])
        except (json.JSONDecodeError, TypeError):
            dropped += 1
            continue

        sub = build_subgraph(packages, bundle, k=k)
        if sub is None:
            dropped += 1
            continue

        sub.y = torch.tensor([int(row["label"])], dtype=torch.float)
        subgraphs.append(sub)
        feats_rows.append(hand_crafted_features(packages, bundle, k=k))
        labels.append(int(row["label"]))
        fc_raw = row.get("failure_category", "")
        if isinstance(fc_raw, float):
            fc_raw = ""
        meta.append({
            "config_id": int(row["config_id"]),
            "source": row.get("source", "unknown"),
            "failure_category": fc_raw or "",
            "n_pkgs": len(packages),
        })

    print(f"materialized {len(subgraphs)} configs; dropped {dropped} (no known packages)")
    return subgraphs, np.stack(feats_rows, axis=0), np.array(labels), meta
