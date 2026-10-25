# Real Laya vs. Einstein Case Routing Benchmark

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

> Empirical, measured benchmark evaluating the real **Laya foundation model** (`convaiinnovations/laya` via pip `laya`), classical **TF-IDF + MLP baseline**, and **Heuristic Rules** against production **Salesforce Einstein Case Classification** on a 1,000-case enterprise CRM dataset.

---

## Limitations & Experimental Scope

> [!WARNING]
> **Read Before Interpreting Results:**
> 1. **Dual Dataset Design (Templated & Group-Held-Out)**: The benchmark evaluates across two distinct datasets: the Original 1,000-case templated benchmark (where train and test share sentence templates, allowing surface n-gram methods like TF-IDF to score 98.50%) and the Group-Held-Out Hard Benchmark (where 10 novel themes are held out with 0 theme overlap, causing TF-IDF to collapse to 30.50%).
> 2. **Full GPU Fine-Tuning Pipeline**: Beyond initial local head-only calibration on Apple Silicon, full transformer encoder fine-tuning of Laya's 421M-parameter ModernBERT-large backbone was conducted on a Google Colab Tesla T4 GPU (`micro_batch=4, grad_accum=16`, effective batch size 64) across 3 distinct seeds (`[2026, 2027, 2028]`), measuring both in-distribution performance and out-of-domain transfer.
> 3. **Unmeasured Einstein Latency**: Salesforce Einstein Case Classification ran asynchronously via batch Case creation inside the Salesforce Sandbox org. Per-case inference latency was not measured; any previous estimates of ~450–800 ms were unmeasured network roundtrip estimates and have been purged.
> 4. **Test Set Scale & Confidence Intervals**: The evaluation set consists of 200 blind cases. All accuracy metrics include Wilson 95% score confidence intervals to reflect statistical uncertainty given this sample size.
> 5. **Scope of Conclusions**: Findings in this repository apply to **zero-shot, head-only calibrated, and full GPU encoder fine-tuned Laya evaluated against classical baselines and Salesforce Einstein** across the documented synthetic benchmarks. Results should be interpreted alongside the training set lexical diversity constraints (~132 unique masked descriptions) rather than generalized uncritically to arbitrary production CRM distributions.

---

## Benchmark Overview

The goal of this benchmark is to measure whether the real Laya foundation model can serve as a local-first, confidence-gated replacement for Salesforce Einstein Case Classification.

* **Dataset**: 1,000 customer cases across 5 balanced departments (`Billing`, `Technical`, `Sales`, `Operations`, `Other`).
  * **Training Closed Cases**: 800 historical closed cases (`data/einstein_benchmark/train_closed.csv`).
  * **Blind Test Cases**: 200 unlabeled incoming cases (`data/einstein_benchmark/test_blind.csv`).
  * **Ground Truth**: `data/einstein_benchmark/test_truth.csv` (80 Easy, 60 Paraphrase, 40 Multi-topic, 20 Ambiguous).
* **Salesforce Einstein Baseline**: Real production model built and activated in a Salesforce Sandbox org, scored on the exact 200 blind cases (`data/einstein_benchmark/einstein_predictions.csv`).
* **Hardware & Environment**: Apple M2 (MacBook Air), 8GB Unified Memory, PyTorch with MPS backend.

---

## Measured Head-to-Head Results

All figures below are dynamically produced by code in this repository or exported directly from the Salesforce Sandbox.

### Table 1: Original Benchmark (Shared Templates, 200 Test Cases)

| System / Model | Training Status | Cases Routed | Routed Accuracy | Overall Top-1 (All 200 Cases) | ECE | Measured Latency (Mean) | Hardware / Context |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Salesforce Einstein** | Trained on 800 closed cases (original) | 173/200 | **99.42%** [96.8%, 99.9%] | **86.00%** [80.5%, 90.1%] | 0.0812 | Unmeasured | Cloud SaaS API (27 unrouted) |
| **Simple Local Baseline (TF-IDF)** | Trained on original closed cases | 173/200 | **100.00%** [97.8%, 100.0%] | **98.50%** [95.7%, 99.5%] | 0.1506 | 0.04 ms | Local Apple M2 CPU (template overlap) |
| **Trained Keyword Rules Baseline** | Tuned from training cases | 173/200 | **60.12%** [52.7%, 67.1%] | **55.50%** [48.6%, 62.2%] | 0.0410 | 0.05 ms | Local Apple M2 CPU (`src/rules_trained.py`, commit `ff8b315`) |
| **Laya English Zero-Shot (Dict)** | No training (clean criteria) | 173/200 | **63.01%** [55.6%, 69.8%] | **56.00%** [49.1%, 62.7%] | 0.2458 | 80.9 ms | Local Apple M2 MPS (`convaiinnovations/laya`, 421M params) |
| **Laya Multilingual Zero-Shot (Dict)** | No training (clean criteria) | 173/200 | **61.27%** [53.8%, 68.2%] | **55.50%** [48.6%, 62.2%] | 0.2811 | 37.9 ms | Local Apple M2 MPS (`convaiinnovations/laya-multilingual`, 322M params) |
| **Laya English Zero-Shot (List)** | No training (labels only) | 173/200 | **48.55%** [41.2%, 56.0%] | **46.50%** [39.7%, 53.4%] | 0.1533 | 57.5 ms | Local Apple M2 MPS (`convaiinnovations/laya`, 421M params) |
| **Laya Multilingual Zero-Shot (List)** | No training (labels only) | 173/200 | **48.55%** [41.2%, 56.0%] | **45.50%** [38.8%, 52.4%] | 0.2230 | 28.7 ms | Local Apple M2 MPS (`convaiinnovations/laya-multilingual`, 322M params) |
| **Laya Full GPU Fine-Tuned** | Option A (Hard 800 train, 3 seeds) | 140/200 | **46.07%** (conf ≥ 0.99)<br><sub>**45.75%** (conf ≥ 0.90)</sub> | **45.50% ± 8.13%**<br><sub>(51.0%, 51.5%, 34.0%)</sub> | 0.4791 | 39.5 ms | Colab Tesla T4 (out-of-domain transfer) |
| **TF-IDF + MLP (Control Baseline)** | Option A (hard closed cases) | 173/200 | **43.35%** [36.2%, 50.8%] | **42.00%** [35.4%, 48.9%] | 0.1506 | 0.04 ms | Local Apple M2 CPU |
| **Untuned Legacy Rules Baseline** | No training | 173/200 | **20.81%** [15.4%, 27.5%] | **22.00%** [16.8%, 28.2%] | 0.3800 | 0.07 ms | Local Apple M2 CPU (`src/rules.py`) |

---

### Table 2: Group-Held-Out Hard Benchmark (Zero Theme Overlap, 200 Test Cases)

| System / Model | Training Status | Cases Routed | Routed Accuracy | Overall Top-1 (All 200 Cases) | ECE | Measured Latency (Mean) | Hardware / Context |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Laya Full GPU Fine-Tuned** | Hard 800 train (mean 3 seeds) | 164/200 (82.0% cov) | **99.83%** (conf ≥ 0.99)<br><sub>**99.83%** (conf ≥ 0.90)</sub> | **98.67% ± 1.25%**<br><sub>(97.0%, 100.0%, 99.0%)</sub> | 0.0393 | 39.5 ms | Colab Tesla T4 (`convaiinnovations/laya`, 421M params) |
| **Laya Multilingual Zero-Shot (Dict)** | No training (clean criteria) | 173/200 | **58.38%** [50.9%, 65.5%] | **54.00%** [47.1%, 60.8%] | 0.2649 | 44.3 ms | Local Apple M2 MPS (`convaiinnovations/laya-multilingual`, 322M params) |
| **Laya Multilingual Zero-Shot (List)** | No training (labels only) | 173/200 | **50.87%** [43.5%, 58.2%] | **47.50%** [40.7%, 54.4%] | 0.1984 | 30.6 ms | Local Apple M2 MPS (`convaiinnovations/laya-multilingual`, 322M params) |
| **Laya English Zero-Shot (List)** | No training (labels only) | 173/200 | **50.87%** [43.5%, 58.2%] | **47.00%** [40.2%, 53.9%] | 0.1614 | 53.8 ms | Local Apple M2 MPS (`convaiinnovations/laya`, 421M params) |
| **Laya English Zero-Shot (Dict)** | No training (clean criteria) | 173/200 | **47.98%** [40.7%, 55.4%] | **46.00%** [39.3%, 52.9%] | 0.2872 | 74.8 ms | Local Apple M2 MPS (`convaiinnovations/laya`, 421M params) |
| **Trained Keyword Rules Baseline** | Tuned from hard training cases | 173/200 | **43.93%** [36.8%, 51.4%] | **41.00%** [34.4%, 47.9%] | 0.0410 | 0.05 ms | Local Apple M2 CPU (`src/rules_trained.py`, commit `ff8b315`) |
| **TF-IDF + MLP (Control Baseline)** | Option A (hard closed cases) | 173/200 | **32.37%** [25.8%, 39.7%] | **30.50%** [24.5%, 37.3%] | 0.1506 | 0.04 ms | Local Apple M2 CPU (collapsed: zero memorized n-grams) |
| **Salesforce Einstein** | Retrained on Hard 800 | 147/200 (73.5% cov) | **27.21%** [20.7%, 34.9%] | **20.00%** [15.0%, 26.1%] | N/A | Unmeasured | Production Sandbox Model (threshold 90, 53 unrouted) |
| **Untuned Legacy Rules Baseline** | No training | 173/200 | **23.70%** [17.9%, 30.7%] | **23.50%** [18.2%, 29.8%] | 0.3650 | 0.07 ms | Local Apple M2 CPU (`src/rules.py`) |

> **Plain English System Determination:**
> - **Laya Full GPU Fine-Tuned vs. Zero-Shot:** Full transformer encoder fine-tuning on a GPU elevated Laya's accuracy on the hard benchmark from 46.0%–54.0% (zero-shot) to **98.67% ± 1.25%** overall and **99.83%** with confidence gating ($\tau \ge 0.90$).
> - **vs. Einstein Sandbox Model:** On the hard test set, with this training data, retrained Salesforce Einstein routed 147 cases exclusively to `Billing`, achieving **27.21% routed accuracy** (40/147) and **20.00% overall accuracy** (40/200), with 53 cases unrouted. On the hard test set, with this training data, Laya GPU Fine-Tuned strongly outperformed Einstein.
> - **Dataset Caveat (132 Unique Descriptions):** The 800 training cases contain only **~132 unique descriptions** after masking digits (~20 themes $\times$ ~40 near-copies). While deep transformer fine-tuning excels at learning these thematic patterns (yielding ~98.7% on the hard test set), it transferred at **45.50%** onto the original templated test set; this synthetic redundancy likely explains the high in-distribution score and suggests that fine-tuning was specialized to the training distribution's vocabulary.
> - **vs. TF-IDF Control:** On the hard benchmark test set, Laya Zero-Shot (**46.0%–54.0%**) and Fine-Tuned (**98.67%**) both beat TF-IDF (**30.50%**), scoped strictly to this held-out test distribution. On the original test set, TF-IDF trained on the hard 800 scored **42.00%** versus Laya Fine-Tuned's **45.50%** average (and **56.00%** for Laya zero-shot dict criteria).



---

## Confidence-Gated Routing Curves

By varying the confidence threshold ($\tau$), each model can trade off automation volume against accuracy:

### Simple Local Baseline (TF-IDF + MLP)
* **200 cases routed (100.0% coverage)**: 197 / 200 correct (**98.50%** accuracy, cutoff 0.2123, 0 unrouted)
* **190 cases routed (95.0% coverage)**: 189 / 190 correct (**99.47%** accuracy, cutoff 0.2386, 10 unrouted)
* **173 cases routed (86.5% coverage)**: 173 / 173 correct (**100.00%** accuracy, cutoff 0.2520, 27 unrouted)
* **150 cases routed (75.0% coverage)**: 150 / 150 correct (**100.00%** accuracy, cutoff 0.2658, 50 unrouted)

### Real Laya Zero-Shot (Descriptive Criteria)
* **200 cases routed (100.0% coverage)**: 112 / 200 correct (**56.00%** accuracy, 0 unrouted)
* **190 cases routed (95.0% coverage)**: 111 / 190 correct (**58.42%** accuracy, 10 unrouted)
* **173 cases routed (86.5% coverage)**: 109 / 173 correct (**63.01%** accuracy, 27 unrouted)
* **150 cases routed (75.0% coverage)**: 99 / 150 correct (**66.00%** accuracy, 50 unrouted)
* **100 cases routed (50.0% coverage)**: 71 / 100 correct (**71.00%** accuracy, 100 unrouted)

### Real Laya Zero-Shot (List Criteria)
* **200 cases routed (100.0% coverage)**: 93 / 200 correct (**46.50%** accuracy, 0 unrouted)
* **190 cases routed (95.0% coverage)**: 91 / 190 correct (**47.89%** accuracy, 10 unrouted)
* **173 cases routed (86.5% coverage)**: 84 / 173 correct (**48.55%** accuracy, 27 unrouted)
* **150 cases routed (75.0% coverage)**: 73 / 150 correct (**48.67%** accuracy, 50 unrouted)

### Real Laya Fine-Tuned (Head-Only + Calibration, RLCD)
* **200 cases routed (100.0% coverage)**: 93 / 200 correct (**46.50%** accuracy, cutoff 0.2199, 0 unrouted)
* **190 cases routed (95.0% coverage)**: 91 / 190 correct (**47.89%** accuracy, cutoff 0.2445, 10 unrouted)
* **173 cases routed (86.5% coverage)**: 86 / 173 correct (**49.71%** accuracy, cutoff 0.2600, 27 unrouted)
* **150 cases routed (75.0% coverage)**: 75 / 150 correct (**50.00%** accuracy, cutoff 0.3061, 50 unrouted)
* **100 cases routed (50.0% coverage)**: 55 / 100 correct (**55.00%** accuracy, cutoff 0.3951, 100 unrouted)

---

## Architectural Findings: Dual Benchmark Synthesis

> **Scoped Conclusion:** Evaluated across the **Original Templated Benchmark** and the **Group-Held-Out Hard Benchmark**:

1. **Comparison vs. Einstein**:
   - On the original templated set, Salesforce Einstein achieved **99.42%** accuracy among routed cases (172/173) and **86.00%** overall top-1 accuracy, outperforming Laya Zero-Shot (56.00% dict / 46.50% list).
   - On the hard test set, with this training data, retrained Salesforce Einstein routed 147 cases exclusively to `Billing` (**27.21% routed accuracy**, **20.00% overall accuracy**), whereas Laya Full GPU Fine-Tuned achieved **98.67% ± 1.25%** overall and **99.83%** routed accuracy with confidence gating ($\tau \ge 0.90$).
2. **Comparison vs. Simple Local Baseline (TF-IDF)**:
   - On the original templated set, TF-IDF achieved **98.50%** overall accuracy due to synthetic template overlap.
   - On the hard test set, TF-IDF collapsed to **30.50%**, while Laya Zero-Shot achieved **46.0%–54.0%** and Laya Fine-Tuned achieved **98.67%**, scoped strictly to this held-out test distribution.
   - When trained on the hard 800 and evaluated on the original set, TF-IDF scored **42.00%** versus Laya Fine-Tuned's **45.50%** average.
3. **Fine-Tuning & Generalization Behavior**:
   - Full transformer encoder fine-tuning on a GPU elevated Laya's accuracy on the hard benchmark from 46.0%–54.0% (zero-shot) to **98.67% ± 1.25%** across 3 seeds.
   - However, transferring this fine-tuned model onto the original templated set yielded **45.50% ± 8.13%**. The training set contains only ~132 unique descriptions after masking digits, which likely explains why the transformer mastered the 20 training themes and suggests specialization to the training distribution's vocabulary.

---

## Repository Structure

```
laya-case-routing-poc/
├── data/
│   ├── einstein_benchmark/
│   │   ├── train_closed.csv          # 800 historical closed cases
│   │   ├── test_blind.csv            # 200 blind test cases
│   │   ├── test_truth.csv            # 200 ground truth labels + difficulties
│   │   ├── einstein_predictions.csv  # Real Salesforce Sandbox Einstein export
│   │   └── BENCHMARK_RESULTS.md      # Detailed benchmark documentation
│   └── quarantine_30case_poc/        # Quarantined legacy mock POC files
├── src/
│   ├── rules.py                      # Heuristic regex rules baseline
│   ├── local_tfidf_baseline.py       # Simple Local Baseline (TF-IDF + MLP)
│   ├── laya_baseline.py              # Real Laya zero-shot evaluation
│   ├── laya_finetune.py              # Real Laya fine-tuning pipeline (laya.train)
│   ├── laya_inference.py             # Real Laya inference engine & confidence gating
│   ├── comparison.py                 # Measured benchmark comparison table generator
│   ├── plot_results.py               # Measured benchmark visualization generator
│   └── einstein_benchmark/
│       └── run_full_benchmark.py     # Master benchmark execution script
├── results/                          # Measured benchmark CSVs, curves & plots
├── ARCHITECTURE.md                   # System architecture and measured findings
├── RESULTS.md                        # Full evaluation tables and metrics
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

### 2. Execute Full Measured Benchmark
```bash
python src/einstein_benchmark/run_full_benchmark.py
```
Or to recompile results and charts from saved CSVs:
```bash
make run
```

---

## Author & Contact

**Pavan Chavali**  
[GitHub Profile](https://github.com/pchavali09) | [Repository](https://github.com/pchavali09/laya-case-routing-poc)

Licensed under the [Apache 2.0 License](LICENSE).
