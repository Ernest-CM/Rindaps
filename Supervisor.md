# 5-Minute Supervisor Presentation

**Project:** A Deep Learning–Based Model for Predicting Dependency Version Conflicts in Python Software Packages
**Presenter:** Rindaps Joshua Nanpon Banda (22CD009487)
**Total duration:** ≈5 minutes (≈700 spoken words at 140 wpm)
**Coverage:** Chapters 1 through 5

Each section below has (1) a slide cue describing what should be on screen and (2) the spoken script. Print this, read it twice, then deliver from the slide cues only — don't read from the page. Numbers, anticipated questions, and a pre-meeting checklist are at the bottom.

---

## SLIDE 1 — Title — 20 seconds

**On screen:** Project title, your name, matriculation number, department, supervisor, date.

> "Good [morning / afternoon], sir. My project is *A Deep Learning–Based Model for Predicting Dependency Version Conflicts in Python Software Packages*. In the next five minutes I'll walk you through the problem, what I built, what I found, and where it goes next."

Then breathe. One second of silence before Slide 2.

---

## SLIDE 2 — Chapter 1: The Problem — 40 seconds

**On screen:** A short `requirements.txt` snippet on the left. On the right, a red `pip install` error reading *"ResolutionImpossible: incompatible dependencies"*.

> "Every Python developer has hit *dependency hell* — `pip install` fails because two packages need incompatible versions of a third. Pip's resolver tells you *after* it fails. There's no warning. The cost is real: failed CI builds, broken deployments, hours of debugging.
>
> My research question is whether a deep learning model can **predict** that a `requirements.txt` will fail, *before* the developer runs `pip install`. I scoped the project to PyPI — the Python ecosystem only — and I framed it as a binary classification problem: given a configuration of packages, predict *compatible* or *conflict*."

**Key point to make crystal clear:** prediction, not resolution. The deliverable is a *risk score*, not an automated fix.

---

## SLIDE 3 — Chapter 2: The Gap in the Literature — 30 seconds

**On screen:** A two-column comparison. Left column: "Rule-based" with Abate 2020 (SAT solvers) and pip's resolver listed. Right column: "Deep learning in software engineering" with Li 2022 (vulnerability detection) and Zhang 2023 (defect prediction) listed. Bottom row spanning both columns: highlighted in your accent colour — "*This project: GNN for dependency conflicts*".

> "The literature splits in two. Rule-based tools like SAT solvers and pip's own backtracker are correct but they don't learn from history. Deep learning in software engineering — graph neural networks — has been used for vulnerability detection and defect prediction, but no published work has applied them specifically to dependency conflicts. That's the gap I'm filling."

Don't recite more references unless asked. Two examples per side is enough.

---

## SLIDE 4 — Chapter 3: System Design — 70 seconds

**On screen:** The six-stage pipeline diagram (from `revised.md` section 3.6.1). Boxes labelled: 1) Data Collection → 2) Graph Construction → 3) Label Generation → 4) Training → 5) Evaluation → 6) CLI. Underneath, three pinned facts: "8,064 packages scraped", "7,699 nodes / 20,910 edges", "17,000 configurations labelled".

> "I implemented the system as a six-stage pipeline. I'll hit the key choices.
>
> Stage One: I scraped PyPI metadata for 8,064 packages. Stage Two: I built a directed dependency graph — 7,699 nodes, 20,910 edges, eight features per node.
>
> Stage Three is the heart of the project. I generated **17,000 dependency configurations** by sampling from popular packages — some pinned to their latest versions, some adversarially perturbed to older versions to manufacture conflicts. I then labelled each one automatically by running `pip install --dry-run` and recording whether it succeeded or failed. The whole labelling pipeline is scripted, parallelised across four workers, and ran in about ten hours unattended.
>
> Stage Four: I trained four models on the same train/validation/test split — two engineered-feature baselines, Random Forest and SVM on an 18-dimensional hand-crafted feature vector, and two graph neural networks, a GCN and a GAT operating on the 2-hop subgraph around each configuration."

If asked *why* GNNs: dependency graphs are naturally graph-structured, and GNNs let the model learn from relationships, not just package names.

---

## SLIDE 5 — Chapter 4: Headline Results — 60 seconds

**On screen:** Table 4.2 from your thesis — four rows (RF, SVM, GCN, GAT), columns AUC, PR-AUC, F1, Precision, Recall. Bold the best in each column. Highlight the GAT and SVM rows.

> "Here are the headline numbers on the held-out test set, with each model evaluated at its validation-tuned threshold.
>
> Random Forest scores ROC-AUC **0.691** and F1 **0.292**.
> SVM scores ROC-AUC 0.688 and F1 **0.303** — the best F1.
> The GCN scores 0.633 and 0.253.
> The GAT scores **0.658** and 0.287.
>
> So the engineered-feature baselines narrowly beat the graph neural networks on global metrics. The GAT is the strongest GNN variant and matches the Random Forest on F1, but it falls about three AUC points short."

Slow down on the numbers. Don't rush this slide.

---

## SLIDE 6 — Chapter 4: Honest Interpretation — 50 seconds

**On screen:** Two bullets. First — "Why the GNN doesn't dominate" with two sub-points: *535 training conflicts × 7,699-package vocabulary* and *engineered features already encode version-pin signal*. Second — "Where the GNN wins" beside a small bar chart showing recall: GAT 0.561, RF 0.325.

> "I report this as a finding, not a disappointment, and I'll explain why on two sides.
>
> Why the GNN doesn't dominate. First, after filtering out failures caused by missing Python 3.11 wheels — which depend on the build environment, not on the graph — I was left with only 535 conflict examples for training across 7,699 packages. That's too sparse for the per-package embeddings to generalise. Second, my hand-crafted features include version-pin information that captures most of the signal directly, so a tree-based model picks it up easily. This pattern is documented in Errica and colleagues' 2020 work showing engineered features remain competitive on small graph-classification benchmarks.
>
> But here's where the GNN *does* win — recall. The GAT catches **56%** of true conflicts; the Random Forest catches **32%**. The GNN trades precision for recall: it's noisier, but it misses fewer real conflicts. For a developer-facing early-warning tool, that's the right operating point — which is why my CLI defaults to the GAT."

This is the most important slide. Practice it until it flows. Examiners will respect honesty here more than spin.

---

## SLIDE 7 — Chapter 4: Operational Artefact — 30 seconds

**On screen:** A terminal screenshot showing `python -m src.cli examples/bad_requirements.txt` and its output — "Conflict probability: 0.81 (HIGH risk)" — with file paths to `examples/good_requirements.txt` and `examples/bad_requirements.txt` underneath.

> "The system ends in a working command-line tool, `conflict-predict`. You hand it a `requirements.txt`, and it returns a conflict probability and a HIGH/MEDIUM/LOW risk band in under two seconds. It runs entirely offline against the cached graph and the trained checkpoint — no PyPI calls at inference time, which means it's fast enough to integrate as a pre-commit hook."

If supervisor asks for a live demo, alt-tab to your terminal. Have it queued up.

---

## SLIDE 8 — Chapter 5: Contribution and Future Work — 50 seconds

**On screen:** Three bullets — "Reproducible 17k-configuration dataset", "First systematic GNN vs engineered-feature comparison on this task", "Working `conflict-predict` CLI". Below, three future-work items in smaller text.

> "To close, three concrete contributions. First, a reproducible labelled dataset of 17,000 dependency configurations, generated by direct install simulation rather than mined from logs. Second, the first systematic comparison of GNNs against engineered-feature baselines on this specific task. Third, a working developer-facing tool.
>
> Three things would push the numbers further. One: mining real `requirements.txt` files from public GitHub repositories to expand and diversify the conflict examples. Two: scraping per-version metadata to enable richer node features. Three: a hybrid model that concatenates the engineered features with the GAT's learned representation — these signals are complementary."

---

## Closing — 10 seconds

> "That's the project, sir. I'm happy to take any questions."

Then stop talking. Don't fill silence. Let your supervisor ask the first question.

---

## Numbers to memorise

Drill these the night before. You will be asked at least three of them.

| Fact | Number |
|---|---|
| Packages scraped | **8,064** |
| Graph nodes / edges | **7,699 / 20,910** |
| Configurations labelled (raw) | **17,000** |
| Configurations after filter | **6,858** |
| Conflict examples in training partition | **535** |
| Class imbalance (test) | **89 / 11** (compatible / conflict) |
| Random Forest test ROC-AUC | **0.691** |
| SVM test F1 | **0.303** |
| GAT test ROC-AUC | **0.658** |
| GAT test recall | **0.561** |
| Random Forest test recall | **0.325** |

---

## Anticipated questions and prepared answers

**Q: "Why didn't the GNN beat the baselines?"**
A: Two reasons. (i) Only 535 training conflicts spread over 7,699 packages — per-package embeddings receive too few gradient updates to generalise. (ii) The engineered features already include version-pin information (`pin_position`, `n_unknown_pins`, count of "old" pins), which is where most of the signal lives. Errica et al. (2020) documented exactly this pattern across small graph-classification benchmarks.

**Q: "Why filter out the `missing_distribution` configurations?"**
A: They're 60 % of the labelled conflicts but they're caused by the absence of a Python 3.11 wheel for a pinned version — a property of build artefacts, not of the dependency graph. No graph-based model can predict those without per-wheel-availability features, which weren't in scope. I narrowed the deliverable from "any pip install failure" to "graph-resolvable conflict" and stated that explicitly in Chapter 4.

**Q: "Why didn't you implement the LSTM you proposed in the abstract?"**
A: The version-sequence signal the LSTM was meant to capture is more efficiently represented by four per-node pin features — in-config flag, is-pinned flag, normalised pin position in release history, and unknown-version flag. The GNN already consumes these directly. The departure from the original methodology is documented in `revised.md`.

**Q: "How real-world are these conflicts?"**
A: The *adversarial* source (5,000 of 17,000 configurations) is synthetic by design — that's where most of the manufactured conflicts come from. The *topn* source (12,000 of 17,000) samples from real popularity weights but doesn't reflect any specific repository's actual `requirements.txt`. A future iteration would mine real requirements files from GitHub — I documented this as future work because it was outside the project timeline.

**Q: "Is this production-ready?"**
A: No. ROC-AUC 0.66 means a non-trivial false-positive rate. It's a research demonstration that establishes a baseline result on a new task. With more data and per-version wheel features, the numbers should move into a useful range. I describe exactly that path in Chapter 5.

**Q: "Why a GAT specifically, not GraphSAGE or R-GCN?"**
A: The attention mechanism in GATv2 produces interpretable per-edge weights — for a developer-facing tool, that means when the model flags a conflict, the attention weights identify which dependencies it considered most influential. GraphSAGE and R-GCN are reasonable next experiments.

**Q: "How long did this take?"**
A: Implementation was roughly [X] weeks. The labelling pipeline alone ran ten hours unattended. Each GNN trains in 5–15 minutes on the laptop GPU.

---

## Don't say — anti-checklist

- ❌ "I didn't get good results." (Wrong framing. You got *meaningful* results.)
- ❌ "The GNN failed." (It didn't fail. It just didn't win.)
- ❌ "I should have done X earlier." (Don't apologise. State findings.)
- ❌ "I'm not sure why." (You know why. Read Section 4.7.2.)
- ❌ "It's only a B.Sc. project." (Never undersell. State what you built.)

---

## Pre-meeting checklist (60 seconds before you walk in)

- [ ] Terminal open with `conda activate rindaps` already done, ready for `python -m src.cli examples\bad_requirements.txt`.
- [ ] [reports/figures/](reports/figures/) folder open in a second window, ready to alt-tab to.
- [ ] [reports/metrics.json](reports/metrics.json) open in a third window — every cited number is in there.
- [ ] Slides open, fullscreen, on Slide 1.
- [ ] Phone silenced.
- [ ] Water glass within reach.
- [ ] Top eleven numbers memorised.
- [ ] Breathing slowed.

Good luck.
