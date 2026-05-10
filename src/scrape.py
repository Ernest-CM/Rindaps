"""
Stage 1 — PyPI metadata scraper.

Downloads JSON metadata for the top-N PyPI packages and their first-level
dependencies, caching every response to data/raw/<package>.json.

Usage:
    python -m src.scrape --top-n 5000 --max-depth 2 --concurrency 20

The script is idempotent. Re-running it skips packages already cached and
resumes where it stopped. Logs go to data/raw/scrape.log and progress is
written to data/raw/index.json on every flush.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import aiohttp

PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"
DEFAULT_RAW_DIR = Path("data/raw")
SEED_FILE = DEFAULT_RAW_DIR / "top-pypi-packages.json"
INDEX_FILE = DEFAULT_RAW_DIR / "index.json"
LOG_FILE = DEFAULT_RAW_DIR / "scrape.log"

# Strip extras and environment markers from a requires_dist line so we can
# extract the bare package name. Example:
#   "requests[security] (>=2.0) ; python_version >= '3.6'"  ->  "requests"
NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")


def _safe_filename(pkg: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", pkg.lower())


def _setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _load_seed(top_n: int) -> list[str]:
    if not SEED_FILE.exists():
        raise FileNotFoundError(
            f"Seed file not found: {SEED_FILE}\n"
            "Download it from "
            "https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json "
            f"and save to {SEED_FILE}"
        )
    with SEED_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("rows") or data.get("packages") or data
    names: list[str] = []
    for row in rows[:top_n]:
        if isinstance(row, dict):
            name = row.get("project") or row.get("name")
        else:
            name = row
        if name:
            names.append(str(name))
    return names


def _extract_dep_names(requires_dist: list[str] | None) -> list[str]:
    if not requires_dist:
        return []
    out = []
    for line in requires_dist:
        m = NAME_RE.match(line)
        if m:
            out.append(m.group(1).lower())
    return out


async def _fetch_one(
    session: aiohttp.ClientSession,
    name: str,
    raw_dir: Path,
    sem: asyncio.Semaphore,
    retries: int = 3,
) -> tuple[str, str, list[str]]:
    """Returns (package_name, status, transitive_deps)."""
    cache_path = raw_dir / f"{_safe_filename(name)}.json"
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                doc = json.load(f)
            deps = _extract_dep_names(doc.get("info", {}).get("requires_dist"))
            return name, "cached", deps
        except (json.JSONDecodeError, OSError):
            cache_path.unlink(missing_ok=True)

    url = PYPI_JSON_URL.format(name=name)
    backoff = 1.0
    for attempt in range(1, retries + 1):
        try:
            async with sem:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 404:
                        logging.warning("404 for %s", name)
                        return name, "missing", []
                    if resp.status >= 500:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history,
                            status=resp.status, message="server error",
                        )
                    if resp.status != 200:
                        logging.warning("HTTP %s for %s", resp.status, name)
                        return name, f"http_{resp.status}", []
                    body = await resp.read()
            cache_path.write_bytes(body)
            doc = json.loads(body)
            deps = _extract_dep_names(doc.get("info", {}).get("requires_dist"))
            return name, "ok", deps
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt == retries:
                logging.error("giving up on %s: %s", name, exc)
                return name, "error", []
            await asyncio.sleep(backoff)
            backoff *= 2
    return name, "error", []


async def _crawl(
    seeds: list[str],
    raw_dir: Path,
    concurrency: int,
    max_depth: int,
) -> dict[str, str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    seen: set[str] = set()
    status: dict[str, str] = {}
    headers = {
        "User-Agent": "rindaps-scraper/1.0 (academic research; PyPI metadata only)",
        "Accept": "application/json",
    }
    connector = aiohttp.TCPConnector(limit=concurrency * 2, ttl_dns_cache=300)

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        frontier = [(s.lower(), 0) for s in seeds]
        wave = 0
        while frontier:
            wave += 1
            batch = [(n, d) for n, d in frontier if n not in seen]
            for n, _ in batch:
                seen.add(n)
            if not batch:
                break
            logging.info("wave %d: fetching %d packages (seen=%d)", wave, len(batch), len(seen))

            tasks = [_fetch_one(session, n, raw_dir, sem) for n, _ in batch]
            next_frontier: list[tuple[str, int]] = []
            done = 0
            t0 = time.time()
            for coro in asyncio.as_completed(tasks):
                name, st, deps = await coro
                status[name] = st
                done += 1
                if done % 100 == 0:
                    rate = done / max(time.time() - t0, 0.001)
                    logging.info("  progress %d/%d (%.1f/s)", done, len(batch), rate)
                # Schedule transitive deps if we haven't hit max depth
                depth = next((d for n2, d in batch if n2 == name), 0)
                if depth + 1 < max_depth:
                    for dep in deps:
                        dep_l = dep.lower()
                        if dep_l not in seen:
                            next_frontier.append((dep_l, depth + 1))
            # Flush index after every wave so re-runs resume cleanly
            INDEX_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")
            frontier = next_frontier
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape PyPI JSON metadata.")
    parser.add_argument("--top-n", type=int, default=5000,
                        help="how many top packages to seed from")
    parser.add_argument("--max-depth", type=int, default=2,
                        help="transitive crawl depth (1=seeds only, 2=+direct deps)")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="max concurrent HTTP requests")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                        help="output directory for cached JSON")
    args = parser.parse_args(argv)

    _setup_logging()
    logging.info("loading seed list (top %d)", args.top_n)
    seeds = _load_seed(args.top_n)
    logging.info("seed count: %d", len(seeds))
    logging.info("starting crawl: depth=%d concurrency=%d", args.max_depth, args.concurrency)

    t0 = time.time()
    status = asyncio.run(_crawl(seeds, args.raw_dir, args.concurrency, args.max_depth))
    elapsed = time.time() - t0

    counts: dict[str, int] = {}
    for st in status.values():
        counts[st] = counts.get(st, 0) + 1
    logging.info("done in %.1fs — totals: %s", elapsed, counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
