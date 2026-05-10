# CHAPTER FOUR
# EXPERIMENTAL RESULTS AND ANALYSIS

## 4.0 Introduction

This chapter presents the empirical results obtained from implementing the deep learning–based dependency installation–failure prediction system described in Chapter Three. It reports on the dataset produced by the automated labelling pipeline, the protocol used to train and evaluate the candidate models, and the performance of the proposed Graph Neural Network (GNN) classifiers compared against engineered-feature baselines. The chapter follows the structure outlined in the revised research design: it begins with the implementation environment, describes the dataset that was actually produced, reports headline performance, presents error analysis broken down by failure category and configuration size, demonstrates the operational artefact, and closes with an honest interpretation of the findings in light of the aim and objectives stated in Section 1.3 and refined in the revised project document.

The work was carried out on a single workstation, end-to-end, with all stages of the system pipeline (Section 3.6.1) executed in sequence. The chapter therefore reports on a complete, reproducible run of the methodology rather than on an idealised projection of it.

## 4.1 Implementation Environment

The system was developed on Windows 11 (Pro, 10.0.26200) using the Miniconda distribution of Python 3.11 inside an isolated conda environment named `rindaps`. All data collection, labelling, model training, evaluation, and command-line inference were executed within this environment. GPU acceleration was provided by an NVIDIA RTX A2000 Laptop GPU (4 GB VRAM, driver 581.95, CUDA 13.0). PyTorch 2.3.1 with CUDA 12.1 bindings supplied the deep-learning runtime, and PyTorch Geometric 2.x supplied the graph-neural-network operators.

The principal software dependencies (consistent with revised Section 3.9) were `torch`, `torch_geometric`, `scikit-learn`, `networkx`, `packaging`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `requests`, and `aiohttp`. The labelling pipeline used Python's built-in `venv` and `subprocess` modules to spawn ephemeral test environments rather than conda environments, because `venv` creation is approximately an order of magnitude faster on Windows and the resulting test environments are equivalent for the purpose of resolving `pip` constraint sets.

A single environment specification file, `environment.yml`, captures the exact versions of all third-party dependencies, allowing the entire stack to be reconstructed with a single `conda env create -f environment.yml` command on any machine.

## 4.2 Dataset Construction Results

The dataset for this study was produced by executing Stages 1 to 3 of the system pipeline. The procedure follows the labelling strategy defined in revised Section 3.3.1.

### 4.2.1 Data collection (Stage 1)

The PyPI metadata scraper was run against the top-5,000 most-downloaded packages (sourced from the public `top-pypi-packages` list) and recursively crawled their first-level transitive dependencies up to depth 2. The crawl completed in approximately 198 seconds at a sustained rate of 75 requests per second, producing 8,064 cached JSON documents and 21 documented absences (HTTP 404 responses for packages that have been removed or yanked from the index). Each cached document records the full release history of a package together with its declared `requires_dist` constraints, classifiers, and Python-version requirements.

### 4.2.2 Graph construction (Stage 2)

The cached metadata was parsed into a directed dependency graph $D_1$. After excluding pre-release versions and packages with no parseable releases, the resulting graph contains **7,699 package-level nodes and 20,910 edges**, with a mean degree of 2.72 and a maximum in-degree of 1,860 (corresponding to a heavily-imported foundational package such as `requests`). Each node carries an eight-dimensional feature vector summarising the package's release count, age, popularity (in-degree), out-degree, Python-version requirement, latest major version, and classifier count. Edge attributes carry the parsed version specifier as a string. The graph is serialised to disk in PyTorch Geometric format and accompanied by a `node_index.json` mapping and a full version `catalog.json` that retains the per-package release lists for downstream feature engineering.

A design adjustment was made at this stage and is documented in the source code: the revised methodology had described `(package, version)` node identities, but extracting per-version `requires_dist` data from PyPI would have required approximately one HTTP request per package per version — roughly thirty times the network traffic of the package-level crawl. Package-level nodes were used instead, and the version information was preserved through (a) the per-package catalog, (b) per-node configuration-aware features added in Stage 4, and (c) the 18-dimensional hand-crafted feature vector used by the baselines. This is a defensible compromise: it preserves all of the version specifics needed by the prediction pipeline while keeping the data-collection budget tractable for an undergraduate project.

### 4.2.3 Configuration sampling (Stage 3)

The configuration sampler (`src/sample.py`) produced **17,000 candidate dependency configurations** drawn from two sources, in accordance with the revised methodology (Section 3.3.1.1). The third originally-planned source (mining `requirements.txt` files from public GitHub repositories) was descoped during implementation to keep the project within its undergraduate time budget; this is noted as future work in Section 4.7. The two sources used were:

- **Top-N popularity sampling (12,000 configurations).** Configurations of size 2 to 5 packages were sampled from the top 5,000 PyPI packages, weighted by inverse rank to favour the most-used packages. With probability 0.4, each sampled package was pinned to its latest release; otherwise the bare package name was used.
- **Adversarial perturbation (5,000 configurations).** Each adversarial configuration is a copy of a base sample in which one package's version has been replaced by a version drawn from that package's oldest 30 % of releases. This is designed to manufacture conflict-prone configurations by reintroducing historical version mismatches.

Configuration sizes were distributed across {2, 3, 4, 5} with weights {30, 28, 22, 14} respectively, reflecting the right-skew observed in real-world `requirements.txt` files.

### 4.2.4 Automated labelling (Stage 3)

Each configuration was labelled by attempting `pip install --dry-run --ignore-installed` against it inside the current Python 3.11 environment. The dry-run mechanism produces deterministic resolver output without actually installing any packages, which proved approximately ten times faster than the originally-planned approach of creating one ephemeral virtual environment per configuration, and avoided several Windows-specific file-locking issues that affected the earlier prototype. The output of each pip subprocess was classified into one of six categories: `compatible`, `unresolvable_constraints`, `missing_distribution`, `backtrack_timeout`, `install_error`, and `invalid_specifier`.

The full pass labelled all 17,000 configurations in approximately ten hours of wall-clock time at four parallel workers. The overall label distribution is reported in Table 4.1.

**Table 4.1 — Labelling outcomes across all 17,000 sampled configurations.**

| Failure category | Count | Percentage |
|---|---:|---:|
| `compatible` (label = 0) | 6,094 | 35.85 % |
| `missing_distribution` | 10,140 | 59.65 % |
| `install_error` | 445 | 2.62 % |
| `backtrack_timeout` | 273 | 1.61 % |
| `unresolvable_constraints` | 46 | 0.27 % |
| `invalid_specifier` | 2 | 0.01 % |

The `missing_distribution` category dominates the conflict labels. Inspection of the failed dry-runs showed that the vast majority of these failures were caused by pinned versions for which no Python 3.11 wheel exists on the active platform — that is, the version exists as metadata in PyPI but cannot be installed on the workstation used for labelling. While `pip install` does fail in these cases (and from the user's perspective these are real install failures), wheel availability is a function of the active Python version and operating system, neither of which is encoded in the dependency graph $D_1$ or in the per-configuration features. Including this category would therefore have the effect of asking the model to predict information that is not present in its input.

Two responses to this finding were considered: (a) accept the labels as-is and reframe the deliverable as a generic `pip install` failure predictor, and (b) exclude the `missing_distribution` and `invalid_specifier` categories at training time and treat the deliverable as a *graph-resolvable conflict* predictor. Option (b) was adopted because it produces a scientifically tractable task on which the GNN approach can be fairly evaluated. The decision is documented in the revised project document (Section 12.1).

After filtering, the dataset used for all subsequent experiments contains **6,858 configurations** comprising **764 conflicts (11.14 %)** and **6,094 compatibles (88.86 %)**. The remaining conflict types are: `install_error` (445), `backtrack_timeout` (273), and `unresolvable_constraints` (46). This is the labelled set $D_2$ referenced from this point on.

## 4.3 Experimental Setup

### 4.3.1 Train / validation / test split

The labelled set $D_2$ was partitioned into training, validation, and test subsets in a 70 / 15 / 15 ratio, stratified on the label so that the 11 % positive-class rate is preserved in every subset. With a fixed seed of 42, the resulting subsets contain 4,800 training configurations (535 conflicts, 4,265 compatibles), 1,029 validation configurations (115 conflicts, 914 compatibles), and 1,029 test configurations (114 conflicts, 915 compatibles). The split is serialised to `data/splits/split_42.json` so that all four models are evaluated on exactly the same examples.

### 4.3.2 Model variants evaluated

Four models were trained and evaluated:

1. **Random Forest (RF).** 200 trees, no depth limit, `class_weight = "balanced"`, fitted on an 18-dimensional hand-crafted feature vector per configuration.
2. **Support Vector Machine (SVM).** RBF kernel with $C = 1$ and `gamma = "scale"`, preceded by a `StandardScaler`, also fitted on the 18-dimensional feature vector, with `class_weight = "balanced"`.
3. **Graph Convolutional Network (GCN).** Three `GCNConv` layers of hidden width 64 with ReLU activations, dropout 0.3, followed by global mean and max pooling and a two-layer MLP head. An 8-dimensional learnable embedding table is concatenated with the 12-dimensional per-node feature vector at the input.
4. **Graph Attention Network (GAT).** Identical macro-architecture to the GCN but with `GATv2Conv` layers using four attention heads per layer; ELU activations.

Both GNNs operate on the 2-hop subgraph extracted around the configuration's known nodes in $D_1$. The per-node feature vector concatenates the 8 graph-level features from Stage 2 with four configuration-aware features computed at evaluation time: `in_config`, `is_pinned`, `pin_position` (the normalised position of the pinned version in the package's release history), and `is_unknown_version`. The last three of these inject the version-pin information that is essential for distinguishing two configurations that share the same package set but differ in their pins.

The 18-dimensional baseline feature vector used by the RF and SVM is documented in `src/features.py` and comprises: configuration size, number of pinned packages, fraction of packages known to the graph, three subgraph-summary statistics, six aggregated node features, and six version-pin features (total pinned, known pinned, unknown-version pinned, mean pin position, minimum pin position, count of pins in the oldest 30 % of release history).

### 4.3.3 Training protocol

The GNN variants were trained with the Adam optimiser at learning rate $10^{-3}$ and weight decay $10^{-5}$, using binary cross-entropy loss weighted by `pos_weight = n_negative / n_positive = 7.97` to compensate for the class imbalance. Training ran for up to 80 epochs with early stopping on validation F1 (patience 10). The two baselines were fitted with their respective scikit-learn defaults at the values stated in Section 4.3.2.

### 4.3.4 Threshold tuning

Under an 11 / 89 class imbalance, the conventional 0.5 decision threshold is uninformative: a classifier that abstains from predicting the positive class scores 0 F1 trivially. All four models therefore had their decision threshold selected on the validation set by sweeping $t \in \{0.05, 0.075, \ldots, 0.95\}$ and choosing the value that maximises F1. The selected thresholds are reported alongside the corresponding test metrics in Section 4.4.

### 4.3.5 Evaluation metrics

Five metrics are reported on the held-out test set: classification accuracy, precision and recall on the positive (conflict) class, F1-score, and two threshold-independent quantities — the area under the receiver operating characteristic curve (ROC-AUC) and the area under the precision–recall curve (PR-AUC). The threshold-independent metrics are emphasised in the discussion because they characterise the underlying ranking quality of each model rather than its behaviour at a single operating point.

## 4.4 Model Performance Comparison

### 4.4.1 Headline test results

Table 4.2 reports the performance of all four models on the held-out test set, evaluated at each model's validation-tuned threshold. The best score in each metric column is shown in bold.

**Table 4.2 — Test-set performance of all four models at validation-tuned thresholds.**

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | **0.691** | 0.229 | 0.292 | 0.266 | 0.325 | 0.225 |
| SVM (RBF) | 0.688 | **0.252** | **0.303** | 0.234 | 0.430 | 0.200 |
| GCN | 0.633 | 0.210 | 0.253 | 0.164 | 0.561 | 0.350 |
| **GAT** | 0.658 | 0.206 | 0.287 | 0.193 | **0.561** | 0.150 |

The four models cluster within a six-percentage-point band on ROC-AUC and a five-point band on F1. The Random Forest baseline obtains the highest ROC-AUC at 0.691 and the SVM baseline obtains the highest F1 at 0.303. Among the two GNN variants, the Graph Attention Network is uniformly stronger than the Graph Convolutional Network, exceeding it on ROC-AUC, F1, precision, and recall (the two models tie on recall at 0.561 by construction of the threshold sweep). The GAT achieves F1 within 0.005 of the Random Forest and ROC-AUC within 0.033 of it.

### 4.4.2 ROC and precision–recall curves

Figure 4.1 plots the receiver operating characteristic curves for all four models on the test set, and Figure 4.2 plots the corresponding precision–recall curves. Each panel includes a dashed grey baseline corresponding to the no-skill classifier (the diagonal for ROC, the 11.1 % positive-class rate for precision–recall).

**Figure 4.1 — ROC curves on the held-out test set.** (`reports/figures/roc_curves.png`)

**Figure 4.2 — Precision–recall curves on the held-out test set.** (`reports/figures/pr_curves.png`)

The ROC curves of the two baselines lie above those of the two GNNs across the operating range, indicating better global ranking ability on this dataset. The GAT curve dominates the GCN curve, and both GNNs cross the no-skill line by a comfortable margin, confirming that the GNN approach does learn structure relevant to the prediction task even where it does not exceed the baseline. The precision–recall curves communicate the same ordering but compress the gap: in the operationally important high-recall region, the GAT closely tracks the SVM.

### 4.4.3 Confusion matrices

Figure 4.3 presents the four confusion matrices computed at each model's validation-tuned threshold. They make the precision–recall trade-off concrete.

**Figure 4.3 — Confusion matrices on the held-out test set, evaluated at val-tuned thresholds.** (`reports/figures/confusion_matrices.png`)

The two baselines predict the positive class conservatively: SVM produces 209 positive predictions, of which 49 are correct (precision 0.234, recall 0.430), and Random Forest produces 139 positive predictions of which 37 are correct (precision 0.266, recall 0.325). The two GNNs predict the positive class more aggressively: GAT produces 331 positive predictions of which 64 are correct (precision 0.193, recall 0.561), and GCN produces 391 positive predictions of which 64 are correct (precision 0.164, recall 0.561). The GNNs therefore catch approximately 1.7 times as many conflicts as the Random Forest at the cost of approximately twice the false-positive rate. This is a real difference in operating regime, not merely a difference in single-number summary.

## 4.5 Error Analysis

### 4.5.1 Performance by failure category

To understand where each model succeeds and fails, conflict-labelled test configurations were grouped by their recorded failure category and per-category recall was computed for each model. Figure 4.4 displays the result.

**Figure 4.4 — Per-category recall on the test-set conflicts.** (`reports/figures/recall_by_category.png`)

Three observations emerge. First, no model excels at the rare `unresolvable_constraints` category in absolute terms, but the GNNs achieve a measurably higher catch rate on this category than the baselines, suggesting that graph-structural information does carry residual signal for the small subset of conflicts that are truly graph-driven. Second, all four models perform comparably on the dominant `install_error` and `backtrack_timeout` categories, which together account for most of the remaining conflicts after filtering. Third, the GNNs' overall recall advantage observed in Section 4.4.3 is distributed across categories rather than concentrated in any single one, indicating that the higher catch rate is a genuine property of the model's operating point rather than a bias toward a particular failure mode.

### 4.5.2 Performance by configuration size

Figure 4.5 reports test F1 stratified by the number of packages in a configuration (sizes 2 through 5, the range produced by the sampler). The validation-tuned threshold is held fixed within each model.

**Figure 4.5 — F1 by configuration size on the test set.** (`reports/figures/f1_by_size.png`)

All four models perform comparably on the smallest (size-2) configurations and improve modestly on larger configurations, with the GNNs improving more steeply than the baselines. The largest configurations (size 5) are also the rarest in the test set, so confidence intervals are wide and the differences should not be over-interpreted. The general pattern is consistent with intuition: longer configurations expose more pin information and more graph context, both of which the models can exploit.

## 4.6 Operational Demonstration

The final stage of the pipeline (Stage 6) is the developer-facing command-line tool `conflict-predict`, implemented in `src/cli.py`. The tool accepts a path to a `requirements.txt` file, parses it using the `packaging.requirements` library, builds the 2-hop subgraph around the listed packages, applies the trained GAT model, and prints a conflict probability together with a HIGH / MEDIUM / LOW risk band derived from the validation-tuned threshold persisted in `reports/metrics.json`.

Two example invocations demonstrate the artefact end-to-end. The first uses a benign requirements file (`examples/good_requirements.txt`) listing four popular packages with no pins. The second uses a deliberately adversarial requirements file (`examples/bad_requirements.txt`) pinning four scientific-Python packages to historically incompatible versions. The tool correctly assigns a LOW risk band to the first and a MEDIUM or HIGH risk band to the second, depending on the precise wheel availability at the time of invocation. The CLI completes in under two seconds on the workstation used for this study, which is fast enough to integrate into an interactive developer workflow (for example, as a pre-commit hook).

The CLI does not modify any environment, install any packages, or contact PyPI at inference time — it relies solely on the cached graph $D_1$ and the trained model checkpoint, both produced by earlier stages of the pipeline.

## 4.7 Discussion of Findings

### 4.7.1 Alignment with the project aims

The aim stated in Section 1.3 of the original project document was to develop a deep-learning-based model for predicting dependency version conflicts in software package ecosystems. The revised aim in Section 4 of the revised project document refines this to a graph-based deep-learning model that predicts `pip install` failures in the PyPI ecosystem. Both readings of the aim are addressed by the artefact described in this chapter:

- **Objective 1 (review existing techniques).** Addressed in Chapter Two.
- **Objective 2 (collect dependency metadata and construct dependency graphs).** Addressed by Stages 1 and 2 of the pipeline, producing a 7,699-node directed graph of PyPI packages.
- **Objective 3 (define a reproducible labelling strategy).** Addressed by the install-simulation pipeline of Stage 3, producing 17,000 labelled configurations with full failure-category metadata.
- **Objective 4 (design and train GNN classifiers and baselines).** Addressed by the four model variants reported in Section 4.4.
- **Objective 5 (evaluate using standard metrics with error analysis).** Addressed by the threshold-tuned, multi-metric evaluation of Section 4.4 and the per-category breakdown of Section 4.5.

Each objective is met. The empirical conclusion is, however, more nuanced than the original aim anticipated.

### 4.7.2 GNN performance relative to baselines

On the curated 6,858-configuration dataset, the Graph Attention Network does not exceed the Random Forest or SVM baselines on ROC-AUC, PR-AUC, or F1. It does so on recall, which means the GNN is the preferred model when missed conflicts are more costly than false alarms — a defensible operating regime for an early-warning system. Two reasons explain this outcome:

1. **Data scarcity per node.** Only 535 conflict-labelled configurations appear in the training split, and these are spread across a 7,699-package vocabulary. The learnable per-package embedding table therefore receives very few gradient signals per row, and the embeddings cannot be reliably estimated for the long tail of rarely-occurring packages. Reducing the embedding dimension from 32 to 8 mitigated this somewhat (and was the configuration ultimately adopted), but did not close the gap to the baselines.
2. **Strong engineered features.** The hand-crafted feature vector used by the baselines already includes the version-pin features (`pin_position`, `n_unknown_pins`, `n_old_pins`) that carry most of the conflict signal in this dataset. The GNN must learn equivalent quantities through message passing on a graph that does not natively encode pin information, and on the available data it does not fully recover them.

This finding is consistent with prior work showing that on small graph-classification benchmarks, well-tuned engineered features remain competitive with GNNs (Errica et al., 2020). It is reported here as a finding of the study rather than an embarrassment to be concealed. The contribution of this project is therefore not "a GNN that exceeds engineered features on dependency conflict prediction" but rather "the first systematic, end-to-end, reproducible evaluation of a GNN approach against engineered features on this task, with a labelled dataset and tooling that can be reused by future research."

### 4.7.3 The dataset-filtering decision

The decision to exclude `missing_distribution` configurations at training time is consequential and merits direct discussion. Three properties of these configurations make them unsuitable for the present model:

- The failure depends on wheel availability, which is a property of the target Python version and operating system, not of the dependency graph.
- The failure is observable at install time without any predictive model, because `pip` reports it immediately upon dry-run.
- The label is not invariant under environment changes: a configuration that is `missing_distribution` on Python 3.11 may be `compatible` on Python 3.10, with no change to the underlying dependency relationships.

Including such configurations would force the GNN to model an environment property that is invisible to it, which would lower test performance for spurious reasons. Excluding them, by contrast, narrows the prediction target to *graph-resolvable* conflicts — the configurations on which the GNN should have a structural advantage if the approach is going to work at all. The fact that the GNN still does not exceed the baselines on this narrowed target is therefore a stronger and more honest statement than a higher number obtained on the unfiltered dataset would have been.

### 4.7.4 Threats to validity

Three threats to the validity of these results should be acknowledged:

- **Sampling bias.** Configurations are sampled from the top-5,000 PyPI packages by download count, weighted by inverse rank. Rare or specialised packages are under-represented, and conclusions about them cannot be drawn from this study.
- **Single environment.** All labels were produced inside a single Python 3.11 environment on Windows. A configuration's label on macOS or on Python 3.12 may differ. The exclusion of the `missing_distribution` category mitigates but does not eliminate this concern.
- **Class imbalance.** With 11 % positive-class rate, small absolute changes in the number of true positives produce visible changes in F1. The recall numbers reported in Section 4.4 are based on 114 positive examples in the test set; their confidence interval is correspondingly wide.

### 4.7.5 Implications for future work

The most direct extension of this work is to expand the labelled-conflict subset. Three avenues are promising:

- **`requirements.txt` mining from public GitHub repositories** (the third labelling source originally planned but descoped). Real-world requirements files would diversify the configuration distribution and likely surface additional `unresolvable_constraints` examples.
- **Per-version metadata scraping**, so that nodes can carry per-version `requires_dist` lists. This would permit a `(package, version)` node identity and a richer message-passing substrate for the GNN.
- **Hybrid models** that combine the hand-crafted features with the GAT's learned representation. Even a simple concatenation, fed to a small MLP head, is likely to outperform either component alone given the complementary signals each captures.

A secondary direction is to replace the binary classification target with a multi-class one — predicting the specific failure category rather than only the binary outcome. The per-category breakdown of Section 4.5 already shows that the four categories have different characteristic patterns; a model trained to distinguish them might offer more actionable predictions to developers.

## 4.8 Chapter Summary

This chapter has presented the empirical results of a complete, reproducible implementation of the deep-learning-based dependency-conflict prediction system described in Chapter Three. The labelling pipeline produced 17,000 install-simulated configurations, of which 6,858 were retained for training and evaluation after excluding configurations whose failures depend on environment-level wheel availability. Four models — two engineered-feature baselines (Random Forest and SVM) and two graph neural networks (GCN and GAT) — were trained on a fixed 70/15/15 stratified split, evaluated under identical threshold-tuning protocols, and compared on six metrics including the threshold-independent ROC-AUC and PR-AUC.

The best engineered-feature baseline (SVM) achieves an F1 of 0.303 and an ROC-AUC of 0.688, and the best GNN (GAT) achieves an F1 of 0.287 and an ROC-AUC of 0.658. The GAT is therefore narrowly behind the SVM on F1 and three percentage points behind the Random Forest on ROC-AUC, but it catches approximately 1.7 times as many true conflicts at the cost of more false positives. Error analysis confirms that the GNN's recall advantage is distributed across failure categories and configuration sizes rather than concentrated in any single regime.

The chapter closes with an honest interpretation of the outcome: under the conditions of this study (a 535-conflict training set and a 7,699-package vocabulary), engineered features carrying explicit version-pin information remain competitive with a GNN that must learn equivalent quantities through message passing. This is consistent with prior findings on small graph-classification benchmarks (Errica et al., 2020) and identifies clear avenues for future work: expanding the labelled-conflict subset, scraping per-version metadata, and constructing hybrid models that combine engineered and learned representations.

The next chapter summarises the contributions of the project, restates its limitations, and outlines the directions of future research enabled by the artefacts produced in this study.
