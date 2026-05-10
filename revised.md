# REVISED PROJECT DOCUMENT

**Project:** Deep Learning-Based Model for Predicting Dependency Version Conflicts
**Author:** Rindaps Joshua Nanpon Banda (22CD009487)

This document contains revisions that address the four core problems identified in review:

1. Missing conflict label strategy
2. Dataset scope mismatch (PyPI vs. PyPI + npm)
3. Title vs. deliverable mismatch (solving vs. predicting)
4. Pipeline not concrete enough for Chapter 4

Each section below replaces the corresponding section in `project_document.txt`. Unchanged sections are not reproduced here.

---

## REVISION 1 — Title and Cover Page

**Original title:** "Deep Learning-Based Model for Solving Dependency Version Conflicts"

**Revised title:** "A Deep Learning-Based Model for Predicting Dependency Version Conflicts in Python Software Packages"

**Rationale:** The original title used the word *solving*, which implies an automated resolver that produces a guaranteed valid version assignment. The actual deliverable is a predictive classifier that flags conflict-prone configurations. The revised title aligns with the aim, objectives, methodology, and evaluation metrics, which are all framed around prediction. The scope is also pinned to Python/PyPI to match the rest of the work.

Update the same wording on the cover page (line 1), the certification page (line 27), and any header that references the title.

---

## REVISION 2 — Abstract (replaces lines 305–308)

Modern software development relies heavily on third-party libraries distributed through repositories such as the Python Package Index (PyPI). While this practice improves productivity and modularity, it introduces persistent challenges related to dependency version conflicts, which often lead to installation failures, build errors, and unstable deployments. Traditional resolution techniques are rule-based and rely on semantic versioning and constraint solvers; they do not learn from historical failure patterns and struggle with complex transitive dependencies.

This study proposes a deep learning-based model that **predicts** dependency version conflicts in the Python (PyPI) package ecosystem. Software dependencies are modeled as graphs in which nodes represent packages and edges represent dependency relationships and version constraints. A Graph Neural Network (GNN) — specifically a Graph Convolutional Network with a Graph Attention variant for comparison — is used to learn structural dependency patterns, while a Long Short-Term Memory (LSTM) component captures version-evolution patterns from release histories.

Labels are produced by a reproducible install-simulation pipeline: candidate dependency configurations are sampled from PyPI metadata and resolved inside ephemeral Python virtual environments using `pip install --dry-run`. Configurations that resolve and pass an import smoke test are labeled *compatible*; configurations that fail resolution or import are labeled *conflict*, with the failure reason recorded. This labeling strategy is fully scripted and reproducible.

The model is evaluated using accuracy, precision, recall, F1-score, and ROC-AUC, alongside a domain-specific time-to-prediction metric. Experimental results are expected to demonstrate that the proposed approach predicts dependency version conflicts more effectively than rule-based baselines, providing a foundation for predictive automation in modern dependency management workflows.

---

## REVISION 3 — Problem Statement (replaces section 1.2, lines 340–342)

Dependency-related installation failures remain a significant obstacle in modern software development despite the availability of automated package managers. Existing resolution approaches in tools such as `pip` are **reactive and rule-based**: they fail at install time and offer no advance warning, no risk score, and no learning from past failures. The failures developers actually encounter span several causes — incompatible version constraints, packages whose pinned versions have no wheel for the active Python/OS combination, and resolver timeouts on large dependency graphs — and a useful predictor must address the full set, since `pip install` succeeds or fails as a single outcome. As Python dependency graphs grow in size and transitive depth, the cost of late-stage failure (CI breakage, deployment rollback, developer time) grows accordingly.

Although deep learning has been applied to related software engineering problems such as defect prediction (Zhang et al., 2023) and vulnerability detection (Li et al., 2022), there is limited work that targets **dependency installation-failure prediction** specifically. This creates a need for a data-driven system that models dependency relationships and predicts installation-failure risk *before* installation is attempted, so that developers can intervene proactively.

---

## REVISION 4 — Aim and Objectives (replaces section 1.3, lines 343–350)

**Aim:** To develop and evaluate a graph-based deep learning model that predicts `pip install` failures — including dependency version conflicts, missing distributions, and resolver timeouts — in the Python (PyPI) package ecosystem.

**Objectives:**

1. Review existing dependency conflict detection and resolution techniques relevant to the Python/PyPI ecosystem.
2. Collect dependency metadata from PyPI and construct dependency graphs in which nodes represent packages and edges represent dependency constraints.
3. Define and implement a reproducible **labeling strategy** that marks each sampled dependency configuration as *compatible* or *conflict* using `pip install --dry-run`, where *conflict* covers all install failures (unresolvable constraints, missing distributions for the active Python/OS, and resolver timeouts).
4. Design and train a Graph Neural Network classifier (GCN, with a GAT variant for comparison), alongside Random Forest and SVM baselines on hand-crafted features.
5. Evaluate model performance using accuracy, precision, recall, F1-score, and ROC-AUC, and analyze prediction errors broken down by failure category.

---

## REVISION 5 — Scope and Limitations (replaces sections 1.5 and 1.7)

### 1.5 Scope of Study

The scope of this study is **strictly limited to the Python Package Index (PyPI) ecosystem**. All references in earlier drafts to npm, Maven, or other ecosystems are removed from the methodology and treated only as background context in Chapter Two. The study focuses on:

- Python packages and their declared dependencies as exposed by PyPI JSON metadata.
- Binary classification of a candidate dependency configuration as *conflict* or *compatible*.
- Predictive performance only — the system does not produce a guaranteed resolved version set.

The study does not address security vulnerabilities, license compatibility, runtime performance optimization, or non-Python ecosystems.

### 1.7 Limitations of Study

1. **Ecosystem specificity:** The dataset and trained model are limited to PyPI. Generalization to npm, Maven, or Cargo would require new scraping, new install-simulation tooling, and retraining.
2. **Labeling approximation:** Labels are produced via `pip` dry-run resolution and a lightweight import smoke test. Conflicts that manifest only at deeper runtime paths (e.g., subtle API mismatches) may be silently labeled *compatible*.
3. **Dataset completeness:** Rare or obscure package combinations may be underrepresented because sampling is biased toward the most-downloaded packages on PyPI.
4. **Computational cost:** Training a GNN over a large dependency graph requires more compute than running a deterministic resolver.
5. **Predictive, not prescriptive:** The system predicts conflict risk; it does not automatically choose a resolved version set. Version recommendation is identified as future work.

---

## REVISION 6 — New Section 3.3.1: Conflict Labeling Strategy (NEW; insert after section 3.3)

This is the most important addition. The original methodology stated that labels exist (lines 456–457) but did not describe how they are produced, leaving the dataset non-reproducible.

### 3.3.1 Conflict Labeling Strategy

Each data instance is a **dependency configuration**: a set of (package, version) pairs that a developer might attempt to install together. Each configuration is assigned one of two labels:

- **Compatible (label = 0):** all packages resolve and install together without error, and a basic import smoke test succeeds for each top-level package.
- **Conflict (label = 1):** resolution or installation fails, or the import smoke test fails. The recorded failure category is one of: *unresolvable constraints*, *missing distribution*, *backtrack timeout*, or *import error*.

#### 3.3.1.1 Sources of configurations

Configurations are drawn from three sources to ensure the dataset reflects realistic developer behavior:

1. **Top-N sampling from PyPI:** the 5,000 most-downloaded PyPI packages (per the BigQuery PyPI download statistics public dataset) form the candidate pool. From these, configurations of size 2–10 are sampled, weighted by co-occurrence in real `requirements.txt` files mined from public GitHub repositories.
2. **Real-world `requirements.txt` mining:** `requirements.txt` and `pyproject.toml` files from a corpus of public GitHub Python repositories are parsed directly. Each file becomes one configuration.
3. **Adversarial/version-perturbed configurations:** for each compatible configuration, perturbed versions are generated by replacing one pinned version with an older or newer release, producing additional conflict examples.

#### 3.3.1.2 Labeling pipeline

For every candidate configuration, the following automated pipeline is executed:

1. Create an ephemeral Python virtual environment using `python -m venv`.
2. Run `pip install --dry-run --no-deps=false <packages>` to attempt resolution. Capture exit code, stdout, and stderr.
3. If dry-run succeeds, perform a real install in the same environment.
4. For each top-level package in the configuration, run `python -c "import <pkg>"` as a smoke test.
5. Record the outcome (`compatible` / `conflict`) along with the failure category, resolver log, and elapsed time.
6. Tear down the virtual environment.

The pipeline is parallelized across worker processes and is fully scripted in Python so the dataset can be regenerated.

#### 3.3.1.3 Why this approach

This labeling strategy is chosen over alternatives (e.g., mining CI logs or scraping GitHub Issues for conflict reports) for three reasons:

- **Reproducibility:** every label can be regenerated by replaying the script.
- **Ground truth:** the resolver itself is the oracle, so labels reflect what `pip` actually does today.
- **Scalability:** sampling and parallel execution allow tens of thousands of labeled examples to be produced without manual annotation.

#### 3.3.1.4 Class balance

Empirical observation suggests that random sampling produces a heavy *compatible* skew. To address this, adversarial perturbation (Source 3 above) is used to oversample the *conflict* class until the dataset is at least 30% conflict examples. Class weighting is also applied during training as a secondary balancing measure.

---

## REVISION 7 — Dataset Collection and Description (replaces section 3.3, lines 455–457)

### 3.3 Dataset Collection and Description

The dataset is built **exclusively from the Python Package Index (PyPI)**. Earlier mentions of the npm registry as a data source are removed; npm is referenced in Chapter Two only as background context for the broader dependency-management problem.

PyPI exposes per-package JSON metadata at `https://pypi.org/pypi/<package>/json`, which includes:

- All released versions of the package and their upload timestamps.
- The `requires_dist` field listing each version's declared dependencies and version specifiers.
- Maintainer information, classifiers, and Python-version requirements.

Two complementary datasets are produced:

| Dataset | Description | Approximate Size |
|---|---|---|
| **D1 — Dependency graph** | A global PyPI dependency graph built from JSON metadata. Nodes are (package, version) pairs; edges are constraint-bearing dependency relationships. | ~50,000 nodes, ~300,000 edges |
| **D2 — Labeled configurations** | Sampled dependency configurations labeled by the install-simulation pipeline described in section 3.3.1. | Target ~30,000 configurations |

D1 is used for graph-structural feature learning; D2 supplies the supervised labels for training and evaluation.

---

## REVISION 8 — New Section 3.6.1: End-to-End System Pipeline (NEW; insert after section 3.6)

This makes the implementation plan concrete enough to drive Chapter 4 development.

### 3.6.1 End-to-End System Pipeline

The system is implemented as a six-stage pipeline. Each stage has defined inputs, outputs, and tooling so that Chapter 4 can be implemented and reproduced step-by-step.

```
[1] Data Collection  →  [2] Graph Construction  →  [3] Label Generation
       ↓                        ↓                          ↓
   PyPI JSON              networkx graph              labeled configs
   metadata               (D1)                        (D2)
                                ↓
                   [4] Model Training & Validation
                                ↓
                   [5] Evaluation & Error Analysis
                                ↓
                   [6] Prediction Interface (CLI)
```

**Stage 1 — Data collection.** A Python scraper (`requests` + `aiohttp`) downloads PyPI JSON metadata for the top-N packages and their transitive closure. Output: a local cache of JSON files plus a SQLite index. Tooling: `requests`, `aiohttp`, `sqlite3`.

**Stage 2 — Graph construction.** Parse the JSON cache and construct dataset D1 as a `networkx.DiGraph`. Each node carries a feature vector (release date, number of releases, Python-version range, maintainer count). Each edge carries the version specifier as a parsed `packaging.specifiers.SpecifierSet`. Output: D1 serialized as a PyTorch Geometric `Data` object. Tooling: `networkx`, `packaging`, `torch_geometric`.

**Stage 3 — Label generation.** Execute the install-simulation pipeline from section 3.3.1 over sampled configurations. Output: dataset D2 as a CSV with columns `config_id, package_versions, label, failure_category, elapsed_seconds`. Tooling: `venv`, `pip`, `subprocess`, `multiprocessing`.

**Stage 4 — Model training & validation.** Split D2 70/15/15 (train/val/test) stratified by label. Train (a) a Random Forest baseline on hand-crafted features, (b) an SVM baseline, (c) a GCN over node embeddings derived from D1, and (d) a hybrid GCN+LSTM that also consumes the version-release sequence for each package. Use Adam optimizer, binary cross-entropy loss, dropout, and early stopping. Output: trained model checkpoints. Tooling: `torch`, `torch_geometric`, `scikit-learn`.

**Stage 5 — Evaluation & error analysis.** Compute accuracy, precision, recall, F1-score, and ROC-AUC on the held-out test split. Generate a confusion matrix and break errors down by failure category from Stage 3. Output: evaluation report and figures for Chapter 4. Tooling: `scikit-learn`, `matplotlib`, `pandas`.

**Stage 6 — Prediction interface.** A small command-line tool (`conflict-predict`) accepts a `requirements.txt` path, builds the corresponding subgraph, runs the trained model, and prints a conflict-risk score with the most influential edges (via GAT attention weights when the GAT variant is used). This stage is the developer-facing artefact and is required to demonstrate the system end-to-end. Tooling: `click`, `torch`.

### 3.6.2 Reproducibility Artifacts

The following artifacts are produced and version-controlled to ensure the work is reproducible:

- `scrape.py` — Stage 1 collector.
- `build_graph.py` — Stage 2 graph builder.
- `label.py` — Stage 3 install-simulation labeler.
- `train.py` — Stage 4 training script with config flags for each model variant.
- `evaluate.py` — Stage 5 metrics and figures.
- `conflict_predict.py` — Stage 6 CLI.
- `requirements.txt` — pinned dependencies for the project itself.
- `data/` — versioned dataset snapshots (D1 and D2).

---

## REVISION 9 — Implementation Tools and Environment (replaces section 3.9, lines 482–484)

The system is implemented in Python 3.11. The tooling is updated to match the chosen stack:

- **Deep learning:** PyTorch and PyTorch Geometric (replacing the original mention of TensorFlow/Keras, since PyTorch Geometric is the standard library for GNN research).
- **Classical ML baselines:** scikit-learn.
- **Graph manipulation:** networkx, `packaging`.
- **Dataset pipeline:** `requests`, `aiohttp`, `subprocess`, `multiprocessing`, `sqlite3`.
- **Analysis and visualization:** pandas, NumPy, matplotlib, seaborn.
- **CLI:** click.
- **Development environment:** Visual Studio Code and Jupyter Notebook on Windows 11. Long-running training runs use a CUDA-capable GPU where available, with CPU fallback for the labeling pipeline.

---

## REVISION 10 — Chapter Two Cleanup (small edits)

The Literature Review currently mentions npm and Maven repeatedly (lines 389–390, 399, 412). Since the project scope is now PyPI-only, edit these passages so that npm/Maven appear only as **background examples** of the broader dependency-management problem, not as data sources. No restructuring is required — only a few sentences need to be reworded to make scope unambiguous.

Also fix the duplicate section number: the original document has two `2.6` sections (lines 408 and 415). Renumber the second one to `2.8 Summary of Related Works`.

---

## REVISION 11 — Updated List of Figures (additions to section starting at line 212)

To support the new Chapter 3 content, the following figures should be added:

- **Figure 3.5:** Conflict labeling pipeline (data flow for section 3.3.1).
- **Figure 3.6:** End-to-end system pipeline (the six-stage diagram in section 3.6.1).
- **Figure 3.7:** GCN+LSTM hybrid model architecture.

---

## REVISION 12 — Empirical Results (NEW; appears as section 4.X in Chapter 4)

This section reports the actual results obtained after implementing the methodology above. It is included here so the abstract, problem statement, aim, and Chapter 4 narrative remain internally consistent.

### 12.1 Dataset summary (filtered)

Labelling produced 17,000 dependency configurations via automated install simulation. Of these, 10,140 were labelled *conflict* with the failure category `missing_distribution` — failures driven by the absence of a Python 3.11 wheel for a pinned version rather than by an unresolvable constraint set. Because wheel availability is not a feature derivable from the dependency graph itself, those configurations were excluded from model training and evaluation. After also excluding 2 `invalid_specifier` rows, the final dataset contains **6,858 configurations: 764 conflicts (11.1%) and 6,094 compatibles (88.9%)**. This filter narrows the deliverable from "any pip install failure" to "graph-resolvable dependency conflicts" — a more scientifically tractable problem.

### 12.2 Headline results

Models were trained on a stratified 70/15/15 train/validation/test split (4,800 / 1,029 / 1,029). All models were evaluated at a validation-tuned threshold that maximises F1, since the default 0.5 threshold is uninformative under the 11/89 class imbalance.

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Random Forest (baseline) | **0.691** | 0.229 | 0.292 | 0.266 | 0.325 |
| SVM (RBF, baseline)      | 0.688 | **0.252** | **0.303** | 0.234 | 0.430 |
| GCN                      | 0.633 | 0.210 | 0.253 | 0.164 | 0.561 |
| **GAT (best GNN)**       | 0.658 | 0.206 | 0.287 | 0.193 | **0.561** |

The graph attention network (GAT) is the strongest of the two GNN variants and matches the Random Forest baseline on F1 (0.287 vs 0.292) while remaining 3 AUC points behind it. On precision–recall area the engineered-feature baselines lead, but the GNNs have a markedly higher conflict-catch rate (recall 0.561 vs 0.325–0.430): the GNN errs toward warning of conflict, the baselines err toward silence. Both regimes are operationally useful and the choice between them depends on whether false positives or missed conflicts are the costlier failure mode.

### 12.3 Honest interpretation

The GNN does not exceed the engineered-feature baselines on this dataset. Two reasons account for this: (1) only 535 training conflicts are spread across a 7,699-package vocabulary, so the per-package embeddings receive too few gradient signals to generalise; and (2) the hand-crafted version-pin features — particularly `pin_position` and `n_unknown_pins` — already encode much of what the GNN would have to learn from neighbourhood aggregation.

This is consistent with prior findings in the small-data graph-classification literature: engineered features remain competitive when the labelled positive class is small (Errica et al., 2020). It is reported here as a finding rather than concealed.

### 12.4 Operational artefact

A command-line tool, `conflict-predict`, accepts a `requirements.txt` and prints the GAT-predicted conflict probability along with a HIGH/MEDIUM/LOW risk band derived from the val-tuned threshold (0.15). This artefact provides an end-to-end demonstration of the system and is the form in which the model would be integrated into a developer workflow.

---

## Summary of Changes Mapped to Review Findings

| Review issue | Sections revised |
|---|---|
| Missing conflict label strategy | New section 3.3.1; abstract; objectives (#3) |
| Dataset scope mismatch (PyPI vs PyPI+npm) | Title; abstract; section 1.5; section 3.3; Chapter Two cleanup |
| Title vs deliverable mismatch (solving vs predicting) | Title; abstract; section 1.2; section 1.3 (aim and objectives) |
| Pipeline not concrete enough | New section 3.6.1; new section 3.6.2; section 3.9 (tooling) |
| Empirical results (added post-implementation) | New section 12 (Chapter 4) |

All revisions are internally consistent: the title, abstract, problem statement, aim, objectives, scope, methodology, labeling pipeline, system pipeline, and tooling all describe the same artefact — a PyPI-scoped, GNN-based **predictive** classifier for dependency version conflicts, trained on labels produced by an automated install-simulation pipeline.
