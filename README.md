# Real Laya vs. Einstein Case Routing Benchmark

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

> Empirical, measured benchmark evaluating the real **Laya foundation model** (`convaiinnovations/laya` via pip `laya`), classical **TF-IDF + MLP baseline**, and **Heuristic Rules** against production **Salesforce Einstein Case Classification** on a 1,000-case enterprise CRM dataset.

---

## Limitations & Experimental Scope

> [!WARNING]
> **Read Before Interpreting Results:**
> 1. **Synthetic Templated Dataset**: The 1,000 cases were generated from hand-authored department templates. Because training (`train_closed.csv`, 800 cases) and test (`test_blind.csv`, 200 cases) share underlying sentence templates and vocabulary, surface n-gram methods (TF-IDF) achieve artificial 98.5% accuracy.
> 2. **Head-Only Fine-Tuning**: Due to Apple Silicon M2 MPS memory constraints (8GB unified memory), Laya's 421M-parameter encoder was kept frozen (`freeze_encoder=True`). Only the linear routing head and temperature scaling were trained. Full backpropagation through the transformer encoder was not performed locally.
> 3. **Unmeasured Einstein Latency**: Salesforce Einstein Case Classification ran asynchronously via batch Case creation inside the Salesforce Sandbox org. Per-case inference latency was not measured; any previous estimates of ~450–800 ms were unmeasured network roundtrip estimates and have been purged.
> 4. **Test Set Scale & Confidence Intervals**: The evaluation set consists of 200 blind cases. All accuracy metrics include Wilson 95% score confidence intervals to reflect statistical uncertainty given this sample size.
> 5. **Scope of Conclusions**: Findings in this repository apply strictly to **zero-shot and head-only fine-tuned Laya on a templated synthetic dataset**, and should not be generalized to full-model fine-tunes or wild customer support distributions.

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

| System / Model | Training Status | Cases Routed | Routed Accuracy @ 86.5% Cov (173/200) | Overall Top-1 (All 200 Cases) | ECE | Measured Latency (Mean) | Hardware / Context |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Salesforce Einstein** | Trained on 800 closed cases (original) | 173/200 | **99.42%** [96.8%, 99.9%] | **86.00%** [80.5%, 90.1%] | 0.0812 | Unmeasured | Cloud SaaS API (27 unrouted) |
| **Trained Keyword Rules Baseline** | Tuned from training cases | 173/200 | **68.21%** [60.9%, 74.7%] | **62.00%** [55.1%, 68.4%] | 0.1650 | 0.05 ms | Local Apple M2 CPU (`src/rules_trained.py`, commit `40f18e9`) |
| **Laya English Zero-Shot (Dict)** | No training (clean criteria) | 173/200 | **63.01%** [55.6%, 69.8%] | **56.00%** [49.1%, 62.7%] | 0.2458 | 80.9 ms | Local Apple M2 MPS (`convaiinnovations/laya`, 421M params) |
| **Laya Multilingual Zero-Shot (Dict)** | No training (clean criteria) | 173/200 | **61.27%** [53.8%, 68.2%] | **55.50%** [48.6%, 62.2%] | 0.2811 | 37.9 ms | Local Apple M2 MPS (`convaiinnovations/laya-multilingual`, 322M params) |
| **Laya English Zero-Shot (List)** | No training (labels only) | 173/200 | **48.55%** [41.2%, 56.0%] | **46.50%** [39.7%, 53.4%] | 0.1533 | 57.5 ms | Local Apple M2 MPS (`convaiinnovations/laya`, 421M params) |
| **Laya Multilingual Zero-Shot (List)** | No training (labels only) | 173/200 | **48.55%** [41.2%, 56.0%] | **45.50%** [38.8%, 52.4%] | 0.2230 | 28.7 ms | Local Apple M2 MPS (`convaiinnovations/laya-multilingual`, 322M params) |
| **TF-IDF + MLP (Control Baseline)** | Option A (hard closed cases) | 173/200 | **46.24%** [39.0%, 53.7%] | **42.00%** [35.4%, 48.9%] | 0.4431 | 0.04 ms | Local Apple M2 CPU |
| **Untuned Legacy Rules Baseline** | No training | 173/200 | **20.81%** [15.4%, 27.5%] | **22.00%** [16.8%, 28.2%] | 0.3800 | 0.07 ms | Local Apple M2 CPU (`src/rules.py`) |

---

### Table 2: Group-Held-Out Hard Benchmark (Zero Theme Overlap, 200 Test Cases)

| System / Model | Training Status | Cases Routed | Routed Accuracy @ 86.5% Cov (173/200) | Overall Top-1 (All 200 Cases) | ECE | Measured Latency (Mean) | Hardware / Context |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Salesforce Einstein** | Trained on 800 closed cases (original) | Pending | *Pending Sandbox Export* | *Pending Sandbox Export* | N/A | Unmeasured | Awaiting Pavan Sandbox Run (zero invented data) |
| **Laya Multilingual Zero-Shot (Dict)** | No training (clean criteria) | 173/200 | **58.38%** [50.9%, 65.5%] | **54.00%** [47.1%, 60.8%] | 0.2649 | 44.3 ms | Local Apple M2 MPS (`convaiinnovations/laya-multilingual`, 322M params) |
| **Trained Keyword Rules Baseline** | Tuned from training cases | 173/200 | **52.60%** [45.2%, 59.9%] | **48.50%** [41.7%, 55.4%] | 0.1850 | 0.05 ms | Local Apple M2 CPU (`src/rules_trained.py`, commit `40f18e9`) |
| **Laya Multilingual Zero-Shot (List)** | No training (labels only) | 173/200 | **50.87%** [43.5%, 58.2%] | **47.50%** [40.7%, 54.4%] | 0.1984 | 30.6 ms | Local Apple M2 MPS (`convaiinnovations/laya-multilingual`, 322M params) |
| **Laya English Zero-Shot (List)** | No training (labels only) | 173/200 | **50.87%** [43.5%, 58.2%] | **47.00%** [40.2%, 53.9%] | 0.1614 | 53.8 ms | Local Apple M2 MPS (`convaiinnovations/laya`, 421M params) |
| **Laya English Zero-Shot (Dict)** | No training (clean criteria) | 173/200 | **47.98%** [40.7%, 55.4%] | **46.00%** [39.3%, 52.9%] | 0.2872 | 74.8 ms | Local Apple M2 MPS (`convaiinnovations/laya`, 421M params) |
| **TF-IDF + MLP (Control Baseline)** | Option A (hard closed cases) | 173/200 | **32.37%** [25.8%, 39.7%] | **30.50%** [24.5%, 37.3%] | 0.4612 | 0.04 ms | Local Apple M2 CPU (collapsed: zero memorized n-grams) |
| **Untuned Legacy Rules Baseline** | No training | 173/200 | **23.70%** [17.9%, 30.7%] | **23.50%** [18.2%, 29.8%] | 0.3650 | 0.07 ms | Local Apple M2 CPU (`src/rules.py`) |

> **Plain English System Determination:**
> - **vs. TF-IDF Control:** Laya Zero-Shot **beat** TF-IDF on the hard benchmark across all configurations (46.0%–54.0% vs. 30.50% top-1; 48.0%–58.4% vs. 32.37% routed), confirming that semantic classification generalizes beyond surface n-gram memorization.
> - **vs. Einstein:** On the original templated set, Laya Zero-Shot achieved 56.00% top-1 vs. Einstein's 86.00%. On the hard benchmark, Einstein evaluation remains **Pending Sandbox Export** until Pavan executes the batch scoring in Salesforce.
> - **Laya English vs. Multilingual:** The two checkpoints show significant prediction divergence across all runs (differing on **24.5% to 32.0%** of test cases). Multilingual leads on the hard set with descriptive criteria (**54.00%** vs. **46.00%**), while English slightly leads on the original set (**56.00%** vs. **55.50%**).


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

## Architectural Findings: Did Real Laya Win?

> **Scoped Conclusion:** Evaluated strictly as **zero-shot and head-only fine-tuned Laya on a templated synthetic dataset**:

1. **Comparison vs. Einstein**: Real Laya **lost** to Salesforce Einstein. Einstein achieved 99.42% accuracy among routed cases (172/173), whereas Laya achieved 48.5% to 63.0% at the same 173-case coverage level.
2. **Comparison vs. Simple Local Baseline**: Real Laya **lost** to the TF-IDF + MLP baseline. The lightweight TF-IDF model achieved 98.50% overall top-1 accuracy and 100.0% accuracy at 173-case coverage in 0.04ms latency, directly leveraging the synthetic n-gram overlap between train and test templates.
3. **Fine-Tuning Behavior**: Fine-tuning Laya's classification head with frozen encoder weights improved probability calibration (ECE dropped from 0.1533 to 0.1095 and multi-class Brier score dropped from 0.7050 to 0.6899), but top-1 accuracy remained at 46.50%. Full encoder fine-tuning requires discrete GPU acceleration and was not tested in this run.

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
├── salesforce/
│   ├── fastapi_bridge.py             # Local webhook bridge for Salesforce (~58ms MPS)
│   ├── CaseLayaRoutingInvocable.cls  # Apex Invocable Action
│   └── README.md                     # Salesforce bridge documentation
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
