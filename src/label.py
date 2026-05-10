"""
Stage 3 — Conflict labeling pipeline.

Reads a CSV of candidate dependency configurations and labels each one as
'compatible' (0) or 'conflict' (1) using `pip install --dry-run --ignore-installed`.

We deliberately skip per-config venv creation. With `--ignore-installed` and
`--dry-run`, pip's resolver re-resolves the constraint set as if nothing were
installed and reports what it would do, without actually modifying state.
This is ~10x faster than spawning a fresh venv per config and avoids the
Windows file-locking issues that hit `shutil.rmtree`. If the resolver result
turns out to be sensitive to the host env, we'll switch to the venv approach.

Input CSV columns:
  config_id (int), packages_json (JSON list of str), source (str)

Output CSV columns (appended):
  config_id, packages_json, label, failure_category, elapsed_seconds,
  source, error_excerpt

Resumable: re-running skips config_ids already in the output CSV.

Usage:
  python -m src.label \
    --input data/labeled/smoke_candidates.csv \
    --output data/labeled/smoke_results.csv \
    --workers 2
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
import sys
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DRYRUN_TIMEOUT = 120  # seconds — past this, pip is almost always backtracking on a doomed config

OUTPUT_FIELDS = [
    "config_id",
    "packages_json",
    "label",
    "failure_category",
    "elapsed_seconds",
    "source",
    "error_excerpt",
]


@dataclass
class LabelResult:
    config_id: int
    packages_json: str
    label: int
    failure_category: str
    elapsed_seconds: float
    source: str
    error_excerpt: str


def _classify_pip_error(text: str) -> str:
    s = (text or "").lower()
    if "no matching distribution" in s or "could not find a version" in s:
        return "missing_distribution"
    if "resolutionimpossible" in s or "conflicting dependencies" in s:
        return "unresolvable_constraints"
    if "is not a valid" in s or "invalidversion" in s:
        return "invalid_specifier"
    return "install_error"


def label_one(args: tuple[int, list[str], str]) -> LabelResult:
    config_id, packages, source = args
    packages_json = json.dumps(packages, ensure_ascii=False)
    t0 = time.monotonic()

    if not packages:
        return LabelResult(config_id, packages_json, 1, "empty_config",
                           time.monotonic() - t0, source, "no packages")

    cmd = [
        sys.executable, "-m", "pip", "install",
        "--dry-run",
        "--ignore-installed",
        "--quiet",
        "--disable-pip-version-check",
        "--no-input",
        *packages,
    ]
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=DRYRUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return LabelResult(config_id, packages_json, 1, "backtrack_timeout",
                           time.monotonic() - t0, source, "dry-run timeout")
    except OSError as exc:
        return LabelResult(config_id, packages_json, 1, "subprocess_error",
                           time.monotonic() - t0, source, str(exc)[:500])

    elapsed = time.monotonic() - t0
    if r.returncode == 0:
        return LabelResult(config_id, packages_json, 0, "", elapsed, source, "")

    err_text = (r.stderr or "") + "\n" + (r.stdout or "")
    return LabelResult(
        config_id, packages_json, 1,
        _classify_pip_error(err_text),
        elapsed, source,
        err_text.strip()[:500],
    )


def load_candidates(path: Path) -> list[tuple[int, list[str], str]]:
    out: list[tuple[int, list[str], str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["config_id"])
            packages = json.loads(row["packages_json"])
            source = row.get("source") or "unknown"
            out.append((cid, packages, source))
    return out


def load_completed(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add(int(row["config_id"]))
            except (KeyError, ValueError):
                continue
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label dependency configurations.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    candidates = load_candidates(args.input)
    done = load_completed(args.output)
    pending = [c for c in candidates if c[0] not in done]
    logging.info(
        "loaded %d candidates; %d already labeled; %d pending; workers=%d",
        len(candidates), len(done), len(pending), args.workers,
    )

    if not pending:
        logging.info("nothing to do")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    file_existed = args.output.exists() and args.output.stat().st_size > 0

    f = args.output.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, quoting=csv.QUOTE_MINIMAL)
    if not file_existed:
        writer.writeheader()
        f.flush()

    pool = None
    n_compat = 0
    n_conflict = 0
    t_start = time.monotonic()
    try:
        if args.workers > 1:
            pool = mp.Pool(processes=args.workers)
            iterator = pool.imap_unordered(label_one, pending, chunksize=1)
        else:
            iterator = (label_one(p) for p in pending)

        for i, result in enumerate(iterator, 1):
            writer.writerow(asdict(result))
            f.flush()
            if result.label == 0:
                n_compat += 1
            else:
                n_conflict += 1
            if i % 5 == 0 or i == len(pending):
                elapsed = time.monotonic() - t_start
                rate = i / max(elapsed, 0.001)
                logging.info(
                    "  %d/%d  compat=%d conflict=%d  %.2f cfg/s",
                    i, len(pending), n_compat, n_conflict, rate,
                )
    except KeyboardInterrupt:
        logging.warning("interrupted; results so far are flushed")
    finally:
        f.close()
        if pool is not None:
            pool.terminate()
            pool.join()

    logging.info(
        "done: total=%d compatible=%d conflict=%d in %.1fs",
        n_compat + n_conflict, n_compat, n_conflict,
        time.monotonic() - t_start,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
