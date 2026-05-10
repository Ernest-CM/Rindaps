# IMPLEMENTATION PLAN

**Project:** A Deep Learning-Based Model for Predicting Dependency Version Conflicts in Python Software Packages
**Owner:** Rindaps Joshua Nanpon Banda (22CD009487)
**Target completion:** End of academic term
**Shell:** Conda (Anaconda Prompt / PowerShell with `conda` activated)

---

## 0. What we are building (one paragraph, no fluff)

A command-line tool, `conflict-predict`, that takes a `requirements.txt` file and outputs a probability that the listed Python packages will fail to install together. Behind the tool is a Graph Neural Network (GCN, with a GAT variant for comparison) trained on a labeled dataset of dependency configurations. Labels are produced automatically by attempting `pip install` of each configuration inside an ephemeral virtual environment and recording success or failure. We compare the GNN against Random Forest and SVM baselines on accuracy, precision, recall, F1, and ROC-AUC. Scope is **PyPI only**. The model **predicts** conflicts; it does not resolve them.

---

## 1. Repository layout

Create this exact structure on day 1. No deviations.

```
Rindaps/
├── data/
│   ├── raw/                  # Cached PyPI JSON files (Stage 1 output)
│   ├── graph/                # Serialized dependency graph D1 (Stage 2 output)
│   ├── labeled/              # Labeled configurations D2 (Stage 3 output)
│   └── splits/               # train/val/test splits (Stage 4 input)
├── src/
│   ├── __init__.py
│   ├── scrape.py             # Stage 1
│   ├── build_graph.py        # Stage 2
│   ├── label.py              # Stage 3
│   ├── features.py           # Node/graph feature extraction
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gcn.py            # Stage 4 — primary model
│   │   ├── gat.py            # Stage 4 — comparison variant
│   │   └── baselines.py      # Stage 4 — RF + SVM
│   ├── train.py              # Stage 4 entrypoint
│   ├── evaluate.py           # Stage 5
│   └── cli.py                # Stage 6 — `conflict-predict` CLI
├── notebooks/
│   ├── 01_explore_pypi.ipynb
│   ├── 02_label_inspect.ipynb
│   └── 03_results.ipynb      # Generates Chapter 4 figures
├── tests/                    # Sanity tests, not unit-test heavy
├── checkpoints/              # Trained model weights
├── reports/
│   ├── figures/              # PNG/PDF for Chapter 4
│   └── metrics.json
├── environment.yml           # Conda env spec
├── requirements.txt          # The project's own pinned deps
├── README.md
└── plan.md                   # This file
```

---

## 2. Environment setup (conda)

Run these in **Anaconda Prompt** (or PowerShell with conda initialized) once, before anything else.

```bash
conda create -n rindaps python=3.11 -y
conda activate rindaps

conda install -c conda-forge numpy pandas scikit-learn matplotlib seaborn jupyter networkx tqdm requests aiohttp click sqlite -y

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install packaging
```

If you have an NVIDIA GPU, replace the torch line with the matching CUDA wheel from pytorch.org. CPU is fine for this project — training is small.

Lock the environment:

```bash
conda env export --no-builds > environment.yml
pip freeze > requirements.txt
```

**Important:** the `rindaps` conda env is for *running our code*. The Stage 3 labeling pipeline uses Python's built-in `venv` to spawn isolated test environments — **not** conda — because `python -m venv` is ~10× faster than `conda create` and we will create tens of thousands of throwaway envs. The `rindaps` env stays clean.

---

## 3. Stage 1 — Data Collection

**Goal:** Download PyPI metadata for the top 5,000 packages and their first-level dependencies. Cache locally as JSON.

**File:** `src/scrape.py`

**Inputs:**
- A static list of top-5000 PyPI packages by download count. Source: [hugovk/top-pypi-packages](https://github.com/hugovk/top-pypi-packages) (download the JSON file, place at `data/raw/top-pypi-packages.json`).

**What the script does:**
1. Reads the top-5000 list.
2. For each package, hits `https://pypi.org/pypi/<package>/json` using `aiohttp` with a concurrency limit of 20.
3. Saves response to `data/raw/<package>.json`.
4. Parses `requires_dist` from each response, collects unique transitive package names, and recursively fetches them (max depth 2 to bound the crawl).
5. Logs progress to `data/raw/scrape.log` and skips packages already cached.

**Run:**

```bash
conda activate rindaps
python -m src.scrape --top-n 5000 --max-depth 2 --concurrency 20
```

**Success criteria:**
- `data/raw/` contains 8,000–15,000 JSON files (top 5k + transitive closure).
- A summary `data/raw/index.json` listing every cached package with version count.
- Total disk: ~500 MB–2 GB.

**Time estimate:** 4–8 hours of network time. Run it overnight on day 1.

**Risks:**
- PyPI rate-limiting: keep concurrency at 20, add a small jitter, retry 5xx.
- Some packages return 404 on edge cases (yanked, namespace conflicts) — log and skip, do not crash.

---

## 4. Stage 2 — Graph Construction

**Goal:** Build the global PyPI dependency graph D1 as a `torch_geometric.data.Data` object.

**File:** `src/build_graph.py`

**Design decisions (do not skip — most students get this wrong):**
- **Nodes are (package, version) pairs**, not packages. Two versions of the same package have different compatibility properties.
- **Edges are directed**: `A@1.2 → B@>=2.0` means "A 1.2 depends on B at version ≥2.0". Edge attributes carry the parsed `SpecifierSet`.
- **Node features (initial vector, length 8):**
  1. Number of releases of the package
  2. Days since first release (normalized)
  3. Days since latest release (normalized)
  4. In-degree (popularity proxy)
  5. Out-degree (dependency count)
  6. Has Python version constraint (0/1)
  7. Major version number of this release
  8. Number of declared classifiers

**What the script does:**
1. Walks `data/raw/*.json`.
2. For each package version, creates a node with the feature vector above.
3. For each `requires_dist` entry, parses with `packaging.requirements.Requirement` and creates edges to every matching version in the graph.
4. Serializes the result to `data/graph/pypi_graph.pt` (PyTorch Geometric format) and `data/graph/pypi_graph.gpickle` (networkx format, for inspection).

**Run:**

```bash
python -m src.build_graph --input data/raw --output data/graph/pypi_graph.pt
```

**Success criteria:**
- ~50,000–150,000 nodes, ~300,000–800,000 edges.
- Graph loads cleanly: `torch_geometric.data.Data.from_dict(torch.load(...))`.
- A printed summary: node count, edge count, feature matrix shape, average degree.

**Time estimate:** 1 day to write, runs in 5–15 minutes.

---

## 5. Stage 3 — Label Generation (the hardest stage)

**Goal:** Produce dataset D2: ~30,000 dependency configurations, each labeled `compatible` (0) or `conflict` (1).

**File:** `src/label.py`

**This is where the project lives or dies.** Get it right early and run it in the background while you do everything else.

**Sources of configurations (mirroring revised.md §3.3.1.1):**

1. **Real `requirements.txt` mining** — clone or fetch ~500 popular Python repos from GitHub, extract their `requirements.txt` and `pyproject.toml`. Use the GitHub Search API or a precomputed list (e.g., the [py-package-dataset](https://github.com/github/CodeSearchNet) corpus). **Target: 5,000 configurations.**

2. **Top-N co-occurrence sampling** — for the top 5,000 PyPI packages, sample configurations of size 2–10 weighted by how often pairs appear together in mined `requirements.txt`. **Target: 15,000 configurations.**

3. **Adversarial perturbation** — for each compatible config from sources 1 and 2, randomly replace one pin with a much older or much newer version. **Target: 10,000 configurations** (most will be conflicts, balancing the dataset).

**Labeling pipeline (per configuration):**

```python
# Pseudocode — see src/label.py for real implementation
def label_one(config_id, packages):
    env_dir = f"data/labeled/_envs/{config_id}"
    try:
        subprocess.run(["python", "-m", "venv", env_dir], timeout=30, check=True)
        pip = f"{env_dir}/Scripts/pip.exe"  # Windows
        result = subprocess.run(
            [pip, "install", "--dry-run", *packages],
            capture_output=True, timeout=120
        )
        if result.returncode != 0:
            return ("conflict", classify_failure(result.stderr))

        # Real install + import smoke test
        subprocess.run([pip, "install", *packages], timeout=300, check=True)
        for pkg in packages:
            top = top_level_module_name(pkg)
            r = subprocess.run(
                [f"{env_dir}/Scripts/python.exe", "-c", f"import {top}"],
                capture_output=True, timeout=30
            )
            if r.returncode != 0:
                return ("conflict", "import_error")
        return ("compatible", None)
    except subprocess.TimeoutExpired:
        return ("conflict", "timeout")
    finally:
        shutil.rmtree(env_dir, ignore_errors=True)
```

**Parallelization:** use `multiprocessing.Pool` with `processes=os.cpu_count() // 2`. Each worker handles one config end-to-end. Disk cleanup is **mandatory** — Stage 3 will fill your disk if you don't `rmtree` after every config.

**Failure categories** (recorded as a column for error analysis in Chapter 4):
- `unresolvable_constraints` — pip backtracks and gives up
- `missing_distribution` — version not on PyPI for current Python/OS
- `backtrack_timeout` — pip exceeded the 120s budget
- `import_error` — installed but import fails (subtle conflict)
- `timeout` — venv creation or other timeout

**Run:**

```bash
python -m src.label \
  --sources requirements,topn,adversarial \
  --target 30000 \
  --workers 4 \
  --output data/labeled/configurations.csv
```

**Output schema** (`data/labeled/configurations.csv`):

| column | type | description |
|---|---|---|
| config_id | int | unique id |
| packages_json | str | JSON list of `[package, version]` pairs |
| label | int | 0 = compatible, 1 = conflict |
| failure_category | str | nullable, one of the categories above |
| elapsed_seconds | float | time to label this config |
| source | str | requirements / topn / adversarial |

**Success criteria:**
- ≥30,000 rows.
- ≥30% of rows labeled `conflict` (rebalance via more adversarial perturbation if not).
- A summary report: count per source, count per failure_category, mean elapsed time.

**Time estimate:** 2 days to write and debug. **24–48 hours** to actually run on a laptop. Start it on a Friday, leave it on the weekend.

**Disk warning:** at peak, Stage 3 may use 5–20 GB of temp space across all worker venvs. Run on a drive with ≥50 GB free. Add a `--cleanup-aggressive` flag that also clears the pip cache between configs.

---

## 6. Stage 4 — Model Training

**Goal:** Train a GCN classifier (with a GAT variant) plus Random Forest and SVM baselines.

**Files:** `src/models/gcn.py`, `src/models/gat.py`, `src/models/baselines.py`, `src/train.py`

### 6.1 Input pipeline

For each labeled configuration:
1. Look up each `(package, version)` in graph D1 to get its node id and feature vector. Configs with packages missing from D1 are skipped (log how many).
2. Extract the **k-hop subgraph** (k=2) around the configuration's nodes from D1.
3. Mark configuration nodes with a binary "in_config" indicator added as a 9th node feature.
4. The label of the subgraph is the configuration's label.

This gives us a graph-level binary classification task: **predict whether this subgraph contains a conflict.**

### 6.2 Models

**GCN** (`src/models/gcn.py`):
- 3 GCNConv layers, hidden dim 64, ReLU between.
- Global mean pooling + global max pooling, concatenated.
- 2-layer MLP head → 1 logit.
- Dropout 0.3 between layers.

**GAT** (`src/models/gat.py`):
- Same shape but with `GATv2Conv`, 4 attention heads.

**Baselines** (`src/models/baselines.py`):
- Hand-crafted features per configuration: total node count in subgraph, total edge count, average node degree, count of conflicting version specifiers, package age stats.
- Random Forest (200 trees) and SVM (RBF kernel) from scikit-learn.

### 6.3 Training loop (`src/train.py`)

- Split D2 70/15/15 stratified by label.
- Optimizer: Adam, lr 1e-3, weight decay 1e-5.
- Loss: `BCEWithLogitsLoss` with `pos_weight` set to inverse class frequency.
- Batch size: 32 subgraphs.
- Max epochs: 100 with early stopping (patience 10 on val F1).
- Save best checkpoint to `checkpoints/{model}_best.pt`.

**Run:**

```bash
python -m src.train --model gcn  --epochs 100 --batch-size 32
python -m src.train --model gat  --epochs 100 --batch-size 32
python -m src.train --model rf
python -m src.train --model svm
```

**Success criteria:**
- All four models complete training without OOM.
- GCN/GAT val F1 > Random Forest val F1 by at least 5 points (if not, error analysis time).
- Checkpoints saved.

**Time estimate:** 1 week to write and tune. Each training run on CPU: 30 min – 3 hours.

---

## 7. Stage 5 — Evaluation

**File:** `src/evaluate.py`

**Computes on the held-out test split:**
- Accuracy, precision, recall, F1, ROC-AUC for all four models.
- Confusion matrix per model.
- F1 broken down by `failure_category` (which conflict types do we predict best?).
- F1 broken down by configuration size (does the model degrade on big configs?).
- Top 20 most-confident wrong predictions for manual inspection.

**Run:**

```bash
python -m src.evaluate --models gcn,gat,rf,svm --output reports/metrics.json
```

**Output:** `reports/metrics.json` plus PNG figures in `reports/figures/`:
- `cm_<model>.png` — confusion matrices
- `roc_all.png` — overlaid ROC curves
- `f1_by_category.png` — bar chart
- `f1_by_size.png` — line chart

**Time estimate:** 2 days. Most of this is making the figures look good for Chapter 4.

---

## 8. Stage 6 — CLI

**File:** `src/cli.py`

**Goal:** A working `conflict-predict` command that demonstrates the system end-to-end during your defense.

**Behavior:**

```bash
$ conflict-predict path/to/requirements.txt
Conflict probability: 0.83 (HIGH RISK)
Top contributing packages:
  - tensorflow==2.0.0  (declared 2019, depends on numpy<1.18)
  - numpy==1.24.0       (released 2023, breaks tensorflow 2.0)
Suggested action: review the version of tensorflow.
```

**Implementation:**
1. Parse `requirements.txt` with `packaging.requirements`.
2. For each requirement, pin to latest matching version using cached PyPI metadata.
3. Build the configuration's subgraph from D1.
4. Load the best GCN checkpoint. Forward pass.
5. If using GAT, extract attention weights for "top contributing packages."
6. Print result.

**Time estimate:** 2 days.

---

## 9. Schedule (10 weeks)

| Week | Focus | Output |
|---|---|---|
| 1 | Repo skeleton, conda env, Stage 1 scrape running overnight | `data/raw/` populated |
| 2 | Stage 2 graph build; start writing Chapter 3 sections in parallel | `data/graph/pypi_graph.pt` |
| 3 | Stage 3 labeling pipeline written and debugged on small batch (100 configs) | `label.py` works on toy input |
| 4 | Stage 3 full run in background; start Stage 4 model code | D2 partial, GCN code drafted |
| 5 | Stage 4 GCN + GAT trained; baselines trained | 4 checkpoints |
| 6 | Stage 5 evaluation; iterate on hyperparameters if needed | `reports/metrics.json` |
| 7 | Stage 6 CLI; demo end-to-end | `conflict-predict` works |
| 8 | Chapter 4 write-up: results, figures, error analysis | Chapter 4 draft |
| 9 | Buffer: re-runs, supervisor revisions, polish | — |
| 10 | Defense prep, slide deck, dry runs | Slides |

---

## 10. Definition of done

The project is complete when **all** of these are true:

- [ ] `conda activate rindaps && python -m src.cli example_requirements.txt` prints a probability without errors.
- [ ] `reports/metrics.json` exists with results for GCN, GAT, RF, SVM on the held-out test set.
- [ ] At least one of the GNN variants beats both baselines on F1.
- [ ] `data/labeled/configurations.csv` has ≥30,000 rows with ≥30% conflict labels.
- [ ] Chapter 4 of the report cites three figures generated from `reports/figures/`.
- [ ] The repository runs from a fresh `git clone` + `conda env create -f environment.yml` on a different machine.

---

## 11. What I (Claude) will do for you

When you say "start Stage X," I will write the actual code for that stage's main file as a working draft, with comments only where the *why* is non-obvious. You run it, paste back any errors, and I fix them. I will not write code for stages we haven't started — premature scaffolding rots.

Recommended next step: tell me to **start Stage 1** so we can kick off the overnight scrape today and unblock everything downstream.
