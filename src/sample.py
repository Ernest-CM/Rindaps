"""
Stage 3b — Sample candidate dependency configurations for labeling.

Two sources:
  1. topn         — random 2-10 package configs sampled from the top-5000
                    available packages, weighted toward more-popular packages.
                    ~40% of packages are pinned to their latest release; the
                    rest are bare names. Mostly produces compatibles.
  3. adversarial  — perturbations of source-1 configs: replace one pin with
                    an old (or extreme) version of the same package. Designed
                    to push the label distribution toward conflicts.

Outputs:
  data/labeled/candidates.csv       — full set (~25,000 configs)
  data/labeled/candidates_spot.csv  — small stratified sample (~200) for a
                                      quick validation labeling run before
                                      committing to the full pass.

Usage:
  python -m src.sample
  python -m src.sample --topn-count 15000 --adv-count 10000 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

DEFAULT_OUT = Path("data/labeled/candidates.csv")
DEFAULT_SPOT = Path("data/labeled/candidates_spot.csv")
CATALOG_PATH = Path("data/graph/catalog.json")
TOP_PACKAGES_PATH = Path("data/raw/top-pypi-packages.json")

# Bell-ish distribution skewed toward smaller configs (matches real reqs.txt sizes).
# Sizes 6+ make pip's backtracker explode; we cap at 5 by default for tractable runtime.
SIZE_CHOICES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
SIZE_WEIGHTS = [30, 28, 22, 14, 10, 6, 4, 2, 2]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _load_inputs() -> tuple[dict, list[str]]:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"missing {CATALOG_PATH} — run src.build_graph first")
    if not TOP_PACKAGES_PATH.exists():
        raise FileNotFoundError(f"missing {TOP_PACKAGES_PATH}")

    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        catalog = json.load(f)
    with TOP_PACKAGES_PATH.open("r", encoding="utf-8") as f:
        top_doc = json.load(f)
    rows = top_doc.get("rows") or top_doc.get("packages") or top_doc
    names: list[str] = []
    for r in rows:
        n = r.get("project") if isinstance(r, dict) else r
        if n:
            names.append(str(n).lower())
    available = [n for n in names if n in catalog]
    return catalog, available


def _sample_topn(
    rng: random.Random,
    catalog: dict,
    available: list[str],
    count: int,
    min_size: int = 2,
    max_size: int = 5,
    pin_prob: float = 0.4,
) -> list[list[str]]:
    universe = available[:5000] if len(available) >= 5000 else available
    # Inverse-rank weights — top-1 is heavily favored, decays smoothly.
    weights = [1.0 / (i + 1) ** 0.5 for i in range(len(universe))]

    size_choices = [s for s in SIZE_CHOICES if min_size <= s <= max_size]
    size_weights = [SIZE_WEIGHTS[SIZE_CHOICES.index(s)] for s in size_choices]
    if not size_choices:
        raise ValueError(f"no sizes in range [{min_size}, {max_size}]")

    configs: list[list[str]] = []
    seen_signatures: set[tuple[str, ...]] = set()
    attempts = 0
    while len(configs) < count and attempts < count * 4:
        attempts += 1
        size = rng.choices(size_choices, weights=size_weights, k=1)[0]
        # Sample with weights, then dedupe (collisions skew us toward popular pkgs)
        chosen = rng.choices(universe, weights=weights, k=size * 2)
        chosen = list(dict.fromkeys(chosen))[:size]
        if len(chosen) < 2:
            continue
        signature = tuple(sorted(chosen))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        pkgs: list[str] = []
        for pkg in chosen:
            if rng.random() < pin_prob:
                versions = catalog[pkg]["versions"]
                if versions:
                    pkgs.append(f"{pkg}=={versions[-1]}")
                    continue
            pkgs.append(pkg)
        configs.append(pkgs)
    return configs


def _bare_name(spec: str) -> str:
    """Extract the package name from a requirement spec like 'requests==2.31.0'."""
    for sep in ("==", ">=", "<=", "~=", ">", "<", "!=", "==="):
        if sep in spec:
            return spec.split(sep, 1)[0].strip().lower()
    return spec.strip().lower()


def _sample_adversarial(
    rng: random.Random,
    catalog: dict,
    base_configs: list[list[str]],
    count: int,
) -> list[list[str]]:
    configs: list[list[str]] = []
    attempts = 0
    while len(configs) < count and attempts < count * 5:
        attempts += 1
        if not base_configs:
            break
        base = list(rng.choice(base_configs))
        if not base:
            continue
        idx = rng.randrange(len(base))
        pkg = _bare_name(base[idx])
        if pkg not in catalog:
            continue
        versions = catalog[pkg]["versions"]
        if len(versions) < 4:
            continue
        n = len(versions)
        # 85% of perturbations pull a version from the OLDEST 30% of releases
        # (most likely to clash with modern transitive deps).
        # 15% pull the OLDEST single version (extreme-old probe).
        if rng.random() < 0.85:
            pool = versions[: max(1, int(n * 0.30))]
        else:
            pool = [versions[0]]
        new_ver = rng.choice(pool)
        new_spec = f"{pkg}=={new_ver}"
        if new_spec == base[idx]:
            continue
        new_config = list(base)
        new_config[idx] = new_spec
        configs.append(new_config)
    return configs


def _stratified_spot(
    topn_configs: list[list[str]],
    adv_configs: list[list[str]],
    target: int,
) -> list[tuple[list[str], str]]:
    half = target // 2
    spot = [(c, "topn") for c in topn_configs[:half]]
    spot += [(c, "adversarial") for c in adv_configs[: target - half]]
    return spot


def _write_csv(path: Path, rows: list[tuple[int, list[str], str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["config_id", "packages_json", "source"],
            quoting=csv.QUOTE_MINIMAL,
        )
        w.writeheader()
        for cid, packages, source in rows:
            w.writerow({
                "config_id": cid,
                "packages_json": json.dumps(packages, ensure_ascii=False),
                "source": source,
            })


def _summarize(rows: list[tuple[int, list[str], str]]) -> None:
    by_source = Counter(r[2] for r in rows)
    by_size = Counter(len(r[1]) for r in rows)
    pinned = sum(1 for r in rows if any("==" in p for p in r[1]))
    logging.info("by source: %s", dict(by_source))
    logging.info("by size:   %s", dict(sorted(by_size.items())))
    logging.info("configs with at least one pinned version: %d", pinned)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample candidate dependency configurations.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--spot-output", type=Path, default=DEFAULT_SPOT)
    parser.add_argument("--topn-count", type=int, default=12000)
    parser.add_argument("--adv-count", type=int, default=5000)
    parser.add_argument("--spot-count", type=int, default=200)
    parser.add_argument("--min-size", type=int, default=2)
    parser.add_argument("--max-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pin-prob", type=float, default=0.4,
                        help="fraction of packages in topn configs that get pinned to latest")
    args = parser.parse_args(argv)

    _setup_logging()
    rng = random.Random(args.seed)

    catalog, available = _load_inputs()
    logging.info("catalog packages: %d  top-N available: %d",
                 len(catalog), len(available))
    if len(available) < 1000:
        logging.warning("only %d packages available — sampler will be limited", len(available))

    logging.info("sampling source 1 (topn): target=%d  size range=[%d,%d]",
                 args.topn_count, args.min_size, args.max_size)
    topn = _sample_topn(
        rng, catalog, available, args.topn_count,
        min_size=args.min_size, max_size=args.max_size, pin_prob=args.pin_prob,
    )
    logging.info("  produced %d configs", len(topn))

    logging.info("sampling source 3 (adversarial): target=%d", args.adv_count)
    adv = _sample_adversarial(rng, catalog, topn, args.adv_count)
    logging.info("  produced %d configs", len(adv))

    rows: list[tuple[int, list[str], str]] = []
    cid = 1000
    for cfg in topn:
        rows.append((cid, cfg, "topn"))
        cid += 1
    for cfg in adv:
        rows.append((cid, cfg, "adversarial"))
        cid += 1

    _write_csv(args.output, rows)
    logging.info("wrote %d configs to %s", len(rows), args.output)
    _summarize(rows)

    spot_rows = _stratified_spot(topn, adv, args.spot_count)
    spot_csv = [(900_000 + i, cfg, src) for i, (cfg, src) in enumerate(spot_rows)]
    _write_csv(args.spot_output, spot_csv)
    logging.info("wrote spot sample (%d configs) to %s",
                 len(spot_csv), args.spot_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
