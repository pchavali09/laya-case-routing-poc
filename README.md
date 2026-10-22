# Real Laya vs. Einstein Case Routing Benchmark

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

> Empirical, measured benchmark evaluating the real **Laya model** (`convaiinnovations/laya` via pip `laya`), classical **TF-IDF + MLP baseline**, and **Heuristic Rules** against production **Salesforce Einstein Case Classification** on a 1,000-case enterprise CRM dataset.

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

All numbers below were produced by code in this repository or exported directly from the Salesforce Sandbox.

### 1. Overall Model Comparison

| System / Model | Model Artifact / Architecture | Overall Top-1 Accuracy (All 200 Cases) | Accuracy @ 173 Cases (Matching Einstein Coverage) | Measured Latency (Mean) | Execution Context |
|---|---|:---:|:---:|:---:|---|
| **Salesforce Einstein** | Production Einstein Sandbox Model | 86.0% (172/200 total) | **99.42%** (172/173 routed) | ~450–800 ms | Cloud SaaS API (27 unrouted) |
| **Simple Local Baseline** | TF-IDF (5k n-grams) + 2-layer MLP (hidden=64) | **98.50%** (197/200) | **100.00%** (173/173 routed) | **0.04 ms** | Local CPU |
| **Real Laya Zero-Shot (Dict)** | `convaiinnovations/laya` (Descriptive Criteria) | 56.00% (112/200) | 63.01% (109/173 routed) | 58.1 ms | Local Apple M2 (MPS) |
| **Real Laya Zero-Shot (List)** | `convaiinnovations/laya` (Choice Labels only) | 46.50% (93/200) | 48.55% (84/173 routed) | 58.0 ms | Local Apple M2 (MPS) |
| **Real Laya Fine-Tuned** | `convaiinnovations/laya` (RLCD head on 800 cases) | 46.50% (93/200) | 50.29% (87/173 routed) | 58.0 ms | Local Apple M2 (MPS) |
| **Heuristic Rules Baseline** | Regex keyword rules (`src/rules.py`) | 22.00% (44/200) | 20.81% (36/173 routed) | **0.005 ms** | Local CPU (written for initial POC schema) |

> **Like-for-Like Note:** Salesforce Einstein routed 173 cases and abstained on 27 (leaving them unrouted for human agent triage). Einstein's 27 unrouted cases are **not** counted as incorrect in the coverage-matched comparison; they represent deliberate abstentions.

---

## Confidence-Gated Routing Curves

By varying the confidence threshold ($\tau$), each model can trade off automation volume against accuracy:

### Simple Local Baseline (TF-IDF + MLP)
* **200 cases routed (100.0% coverage)**: 197 / 200 correct (**98.50%** accuracy, 0 unrouted)
* **190 cases routed (95.0% coverage)**: 189 / 190 correct (**99.47%** accuracy, 10 unrouted)
* **173 cases routed (86.5% coverage)**: 173 / 173 correct (**100.00%** accuracy, 27 unrouted)
* **150 cases routed (75.0% coverage)**: 150 / 150 correct (**100.00%** accuracy, 50 unrouted)

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
* **100 cases routed (50.0% coverage)**: 55 / 100 correct (**55.00%** accuracy, 100 unrouted)

---

## Architectural Findings: Did Real Laya Win?

1. **Comparison vs. Einstein**: Real Laya **lost** to Salesforce Einstein. Einstein achieved 99.42% accuracy among routed cases (172/173), whereas Laya achieved 48.5% to 63.0% at the same 173-case coverage level.
2. **Comparison vs. Simple Local Baseline**: Real Laya **lost** to the TF-IDF + MLP baseline. The lightweight TF-IDF model achieved 98.50% overall top-1 accuracy and 100.0% accuracy at 173-case coverage in 0.04ms latency, training in under 1 second.
3. **Fine-Tuning Behavior**: Fine-tuning Laya's classification head on 800 closed cases improved probability calibration (ECE dropped from 0.1533 to 0.1095 and multi-class Brier score dropped from 0.7050 to 0.6899), but top-1 accuracy remained at 46.50% because the transformer encoder representations were frozen.

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
│   └── einstein_benchmark/
│       └── run_full_benchmark.py     # Master benchmark execution script
├── salesforce/
│   ├── fastapi_bridge.py             # Local webhook bridge for Salesforce
│   ├── CaseLayaRoutingInvocable.cls  # Apex Invocable Action
│   └── README.md                     # Salesforce bridge documentation
├── results/                          # Benchmark output tables & summaries
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

### 3. Run Individual Components
* **Real Laya Zero-Shot**:
  ```bash
  python src/laya_baseline.py
  ```
* **Simple Local Baseline (TF-IDF + MLP)**:
  ```bash
  python src/local_tfidf_baseline.py
  ```
* **Real Laya Fine-Tuning**:
  ```bash
  python src/laya_finetune.py
  ```

---

## Author & Contact

**Pavan Chavali**  
[GitHub Profile](https://github.com/pchavali09) | [Repository](https://github.com/pchavali09/laya-case-routing-poc)

Licensed under the [Apache 2.0 License](LICENSE).
