# Real Laya vs. Einstein Case Routing Benchmark

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

> Empirical, measured benchmark evaluating the real **Laya foundation model** (`convaiinnovations/laya` via pip `laya`), classical **TF-IDF + MLP baseline**, and **Keyword Rules** against **Salesforce Einstein Case Classification** on an enterprise CRM dataset.

> **Credits:** Laya is an open-source decision model by NandhaKishorM (Apache 2.0, [convaiinnovations/laya on Hugging Face](https://huggingface.co/convaiinnovations/laya)). I built the benchmark harness, not the model.

---

## Benchmark Setup & Experimental Design

The goal of this benchmark is to measure whether the real Laya foundation model can serve as a local-first, calibrated decision model for enterprise customer support ticket routing compared to classical local baselines and cloud SaaS AI.

* **Single Test Set**: The 200 "hard" cases (`data/einstein_hard_benchmark/test_blind.csv` and `data/einstein_hard_benchmark/test_truth.csv`), consisting of 10 business themes completely absent from the training set (0 shared themes, verified by a 3-gram Jaccard index of 0.52% and zero identical descriptions).
* **Training Set**: The 800 narrow cases (`data/einstein_hard_benchmark/train_closed.csv`), distributed across 20 business themes (40 cases per theme). After masking numerical digits (`\d+ -> <NUM>`), the training corpus collapses into **about 132 distinct descriptions**.
* **Departments**: 5 balanced departments: `Billing`, `Technical`, `Sales`, `Operations`, `Other` (160 training cases and 40 test cases per department).
* **Hardware & Environment**:
  * Local Baselines & Zero-Shot: Apple M2 (MacBook Air), 8GB Unified Memory, PyTorch with MPS backend.
  * GPU Fine-Tuning: Google Colab Tesla T4 GPU (`micro_batch=4, grad_accum=16`, effective batch size 64).
  * Cloud SaaS AI: Salesforce Enterprise Sandbox org running production Einstein Case Classification.

---

## Measured Results (Score on the 200-Case Test)

All figures below are measured directly by code in this repository or exported from the Salesforce Sandbox.

| Model / System | Training Data | Top-1 Accuracy (200-Case Test) | Notes |
|---|---|:---:|---|
| **Laya fine-tuned** | Narrow 800 cases (GPU fine-tune) | **98.7%** | Average of 3 seeds (97.0% / 100.0% / 99.0%) |
| **TF-IDF + MLP** | Broader set of 800 cases | **96.0%** | Scored on the same 200-case test; only training data changed |
| **Laya zero-shot** | None (no fine-tuning) | **46% to 54%** | 46.0%–47.0% (English), 47.5%–54.0% (Multilingual) |
| **Keyword rules** | Built from the 800 training cases only | **41.0%** | Frozen rules tuned strictly on training cases (`src/rules_trained.py`) |
| **TF-IDF + MLP** | Same narrow 800 cases | **30.5%** | Fails when test themes are absent from training (61/200) |
| **Salesforce Einstein Case Classification** | Sandbox model trained on narrow 800 | **20.0%** | 40 of 200 right; 147 routed, all to Billing; 53 left unrouted and counted as not right; 95% CI 15.0% to 26.1% |

> [!NOTE]
> **Key Observations & Contextual Caveats:**
> - **It is one run, in one sandbox, on synthetic data.**
> - **The two TF-IDF rows are the same model on the same test; only the training data changed.** When trained on a broader 800-case distribution, TF-IDF achieved 96.0%; when trained on the narrow 800 cases, it collapsed to 30.5%.
> - **The same fine-tuned Laya models scored 45.5% on a different, earlier test, so 98.7% is not a general number.** The training set consists of 800 narrow cases with about 132 distinct descriptions after masking digits (~20 themes $\times$ ~40 near-copies). This synthetic repetition likely explains why the transformer mastered the 20 training themes (yielding ~98.7% on this test set) and suggests specialization to the training distribution's vocabulary.
> - **Salesforce Einstein Sandbox Outcome:** When retrained on the narrow 800 cases in the sandbox with Automate Value enabled at 90% confidence threshold, Einstein routed 147 cases exclusively to `Billing` (correctly identifying all 40 Billing cases, 27.2% routed accuracy) and left 53 cases unrouted. With unrouted cases counted as not right, overall accuracy was 20.0% (40/200).

---

## Empirical Visualizations

* **[Confidence-Gated Routing Simulation](results/routing_simulation.png)**: Trade-off curves between coverage volume and routed accuracy across thresholds.
* **[Calibrated Confidence Distribution](results/confidence_distribution.png)**: Posterior confidence score distribution comparing zero-shot and GPU fine-tuned Laya.
* **[Accuracy by Difficulty Tier](results/difficulty_breakdown.png)**: Top-1 accuracy across Easy (80), Paraphrase (60), Multi-Topic (40), and Ambiguous (20) cases.

---

## Repository Structure

```
laya-case-routing-poc/
├── data/
│   ├── einstein_benchmark/           # Original benchmark dataset and references
│   └── einstein_hard_benchmark/      # Hard benchmark dataset (800 train, 200 test)
│       ├── train_closed.csv          # 800 closed cases (~132 unique masked descriptions)
│       ├── test_blind.csv            # 200 blind test cases (10 held-out themes)
│       ├── test_truth.csv            # 200 ground truth labels + difficulties
│       ├── train_salesforce_import.csv # Salesforce Data Loader import format
│       ├── test_salesforce_import.csv  # Salesforce Data Loader test import format
│       ├── einstein_hard_raw_export.csv # Raw predictions exported from Salesforce Sandbox
│       ├── EINSTEIN_HARD_RUN_NOTES.md  # Detailed run configuration and audit notes
│       └── EINSTEIN_EXPORT_INSTRUCTIONS.md # Sandbox replication guide
├── notebooks/
│   ├── einstein_benchmark.ipynb      # Local benchmark evaluation notebook
│   └── laya_gpu_full_finetune.ipynb  # Colab GPU full encoder fine-tuning pipeline
├── src/
│   ├── rules_trained.py              # Keyword rules built from 800 training cases only
│   ├── local_tfidf_baseline.py       # TF-IDF + MLP baseline
│   ├── laya_baseline.py              # Real Laya zero-shot evaluation
│   ├── laya_finetune.py              # Local Laya fine-tuning pipeline
│   ├── laya_inference.py             # Inference engine & confidence gating
│   └── einstein_benchmark/
│       ├── score_handoff_results.py  # Master evaluation scoring script
│       ├── run_dual_benchmark.py     # Dual benchmark runner
│       ├── generate_all_baseline_predictions.py # Baseline per-case predictor
│       └── validate.py               # Dataset validation suite
├── results/                          # Committed per-case prediction CSVs & scoring summaries
│   ├── colab_finetune/               # Multi-seed GPU predictions and summary JSON
│   ├── handoff_scoring_summary.json  # Comprehensive evaluation summary
│   └── measured_benchmark_hard_summary.json # Hard benchmark summary metrics
├── ARCHITECTURE.md                   # System architecture and design documentation
├── RESULTS.md                        # Full evaluation tables and metrics
├── pyproject.toml
└── requirements.txt
```

---

## Reproducing the Benchmark

### 1. Environment Setup
```bash
git clone https://github.com/pchavali09/laya-case-routing-poc.git
cd laya-case-routing-poc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Dataset Validation
```bash
python src/einstein_benchmark/validate.py
```

### 3. Score All Models
```bash
python src/einstein_benchmark/score_handoff_results.py
```

---

## Author & Contact

**Pavan Chavali**  
[GitHub Profile](https://github.com/pchavali09) | [Repository](https://github.com/pchavali09/laya-case-routing-poc)

Licensed under the [Apache 2.0 License](LICENSE).
