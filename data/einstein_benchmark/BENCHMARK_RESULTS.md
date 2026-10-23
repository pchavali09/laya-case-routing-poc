# 1,000-Case Einstein Benchmark: Measured Empirical Results

This document records the official measured results on the 1,000-case Einstein benchmark (800 closed training cases, 200 blind test cases).

* **Dataset**: 800 training closed cases (`train_closed.csv`), 200 blind test cases (`test_blind.csv`), 200 ground truth labels (`test_truth.csv`).
* **Salesforce Einstein Baseline**: Real model trained and evaluated in a production Salesforce Sandbox org (`einstein_predictions.csv`).
* **Real Laya Models**: `convaiinnovations/laya` (**421,293,827 parameters**, `answerdotai/ModernBERT-large` backbone 394.8M + 2-layer head 25.2M; pip `laya` v0.4.13 by NandhaKishorM). Sourced directly from checkpoint `rl_agent_config.json` and verified via PyTorch parameter enumeration `sum(p.numel() for p in model.parameters())`.
* **Multilingual Model**: `convaiinnovations/laya-multilingual` (**321,908,995 parameters**, `jhu-clsp/mmBERT-base` backbone).
* **Dual Benchmark Report**: See [`data/einstein_hard_benchmark/DUAL_BENCHMARK_RESULTS.md`](file:///Users/pavanchavali/Documents/SFProjects/sf-claude-test/laya-case-routing-poc/data/einstein_hard_benchmark/DUAL_BENCHMARK_RESULTS.md) for full side-by-side comparison between this original templated set and the group-held-out hard benchmark.
* **Simple Local Baseline**: TF-IDF (5,000 bigrams) + 2-layer MLP (hidden_dim=64, seed 2026).
* **Heuristic Rules Baseline**: Regex keyword rules (`src/rules.py`, written for legacy POC schema).
* **Scope**: Evaluates **zero-shot and head-only fine-tuned Laya on a templated synthetic set**.

---

## 1. Overall Head-to-Head Comparison (200 Blind Cases, with Wilson 95% Confidence Intervals)

| System / Model | Architecture / Artifact | Overall Top-1 Acc (200 Cases) | Wilson 95% CI (Overall) | Acc @ 173 Cases (Matching Einstein) | Wilson 95% CI (Routed) | Measured Latency (Mean) | Status vs. Einstein |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Salesforce Einstein** | Production Einstein Sandbox Model | 86.0% (172/200 total) | [80.5%, 90.1%] | **99.42%** (172/173 routed) | [96.8%, 99.9%] | Unmeasured (batch in SF) | **Reference Standard** (27 unrouted) |
| **Simple Local Baseline** | TF-IDF (5k n-grams) + 2-layer MLP (hidden=64) | **98.50%** (197/200) | [95.7%, 99.5%] | **100.00%** (173/173 routed) | [97.8%, 100.0%] | **0.04 ms** | **Tied / Exceeded Einstein** (n-gram overlap) |
| **Real Laya Zero-Shot (Dict)** | `convaiinnovations/laya` (Descriptive Criteria) | 56.00% (112/200) | [49.1%, 62.7%] | 63.01% (109/173 routed) | [55.6%, 69.8%] | 58.1 ms | **Lost to Einstein & TF-IDF** |
| **Real Laya Zero-Shot (List)** | `convaiinnovations/laya` (Choice Labels only) | 46.50% (93/200) | [39.7%, 53.4%] | 48.55% (84/173 routed) | [41.2%, 56.0%] | 58.0 ms | **Lost to Einstein & TF-IDF** |
| **Real Laya Fine-Tuned** | `convaiinnovations/laya` (Head-only RLCD, ECE=0.1095) | 46.50% (93/200) | [39.7%, 53.4%] | 49.71% (86/173 routed) | [42.3%, 57.1%] | 58.0 ms | **Lost to Einstein & TF-IDF** |
| **Heuristic Rules Baseline** | Regex keyword rules (`src/rules.py`) | 22.00% (44/200) | [16.8%, 28.2%] | 20.81% (36/173 routed) | [15.4%, 27.5%] | **0.073 ms** | **Lost (Keyword mismatch)** |

---

## 2. Breakdown by Difficulty Tier

| Difficulty Tier | Cases | Rules Baseline | Real Laya Zero-Shot (Dict) | Real Laya Zero-Shot (List) | Real Laya Fine-Tuned (Head) | Simple Local Baseline (TF-IDF) | Salesforce Einstein (Routed: 173) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OVERALL (200 Cases)** | **200** | **22.0% (44/200)** | **56.0% (112/200)** | **46.5% (93/200)** | **46.5% (93/200)** | **98.5% (197/200)** | **99.4% (172/173 routed)** |
| `easy` (direct keywords) | 80 | 23.8% (19/80) | 65.0% (52/80) | 58.8% (47/80) | 58.8% (47/80) | **98.8% (79/80)** | 100.0% (75/75 routed, 5 unrouted) |
| `paraphrase` (synonyms & colloquial) | 60 | 20.0% (12/60) | 53.3% (32/60) | 41.7% (25/60) | 41.7% (25/60) | **100.0% (60/60)** | 100.0% (48/48 routed, 12 unrouted) |
| `multi_topic` (cross-department topics) | 40 | 22.5% (9/40) | 37.5% (15/40) | 30.0% (12/40) | 30.0% (12/40) | **95.0% (38/40)** | 97.1% (33/34 routed, 6 unrouted) |
| `ambiguous` (under-specified inquiries) | 20 | 20.0% (4/20) | 65.0% (13/20) | 45.0% (9/20) | 45.0% (9/20) | **100.0% (20/20)** | 100.0% (16/16 routed, 4 unrouted) |

---

## 3. Confidence-Gated Auto-Routing Curves

### Simple Local Baseline (TF-IDF + MLP)
* **200 / 200 routed (100.0%)**: 197 / 200 correct (**98.50%** accuracy, cutoff 0.2123, 0 unrouted)
* **190 / 200 routed (95.0%)**: 189 / 190 correct (**99.47%** accuracy, cutoff 0.2386, 10 unrouted)
* **173 / 200 routed (86.5%, matches Einstein)**: 173 / 173 correct (**100.00%** accuracy, cutoff 0.2520, 27 unrouted)
* **150 / 200 routed (75.0%)**: 150 / 150 correct (**100.00%** accuracy, cutoff 0.2658, 50 unrouted)
* **100 / 200 routed (50.0%)**: 100 / 100 correct (**100.00%** accuracy, cutoff 0.2888, 100 unrouted)

### Real Laya Zero-Shot (Descriptive Dictionary Criteria)
* **200 / 200 routed (100.0%)**: 112 / 200 correct (**56.00%** accuracy, 0 unrouted)
* **190 / 200 routed (95.0%)**: 111 / 190 correct (**58.42%** accuracy, 10 unrouted)
* **173 / 200 routed (86.5%, matches Einstein)**: 109 / 173 correct (**63.01%** accuracy, 27 unrouted)
* **150 / 200 routed (75.0%)**: 99 / 150 correct (**66.00%** accuracy, 50 unrouted)
* **100 / 200 routed (50.0%)**: 71 / 100 correct (**71.00%** accuracy, 100 unrouted)

### Real Laya Zero-Shot (Choice Label List Criteria)
* **200 / 200 routed (100.0%)**: 93 / 200 correct (**46.50%** accuracy, 0 unrouted)
* **190 / 200 routed (95.0%)**: 91 / 190 correct (**47.89%** accuracy, 10 unrouted)
* **173 / 200 routed (86.5%, matches Einstein)**: 84 / 173 correct (**48.55%** accuracy, 27 unrouted)
* **150 / 200 routed (75.0%)**: 73 / 150 correct (**48.67%** accuracy, 50 unrouted)

### Real Laya Fine-Tuned (Head + Temperature Calibration, RLCD)
* **200 / 200 routed (100.0%)**: 93 / 200 correct (**46.50%** accuracy, cutoff 0.2199, 0 unrouted)
* **190 / 200 routed (95.0%)**: 91 / 190 correct (**47.89%** accuracy, cutoff 0.2445, 10 unrouted)
* **173 / 200 routed (86.5%, matches Einstein)**: 86 / 173 correct (**49.71%** accuracy, cutoff 0.2600, 27 unrouted)
* **150 / 200 routed (75.0%)**: 75 / 150 correct (**50.00%** accuracy, cutoff 0.3061, 50 unrouted)
* **100 / 200 routed (50.0%)**: 55 / 100 correct (**55.00%** accuracy, cutoff 0.3951, 100 unrouted)

---

## 4. Key Findings Summary

> **Scoped Statement:** All conclusions below apply strictly to **zero-shot and head-only fine-tuned Laya on a templated synthetic set**:

1. **Real Laya vs. Einstein**: Real Laya **lost** to Salesforce Einstein (63.0% zero-shot dict / 49.7% fine-tuned vs. 99.4% Einstein at matching 173-case coverage).
2. **Real Laya vs. TF-IDF**: Real Laya **lost** to the Simple Local Baseline (TF-IDF + MLP), which reached 98.5% top-1 accuracy in 0.04 ms due to synthetic n-gram template overlap between train and test sets.
3. **Abstentions**: Salesforce Einstein correctly abstained on 27 cases (12 paraphrase, 6 multi-topic, 5 easy, 4 ambiguous), avoiding misclassifications on uncertain tickets.
4. **Calibration**: While head fine-tuning did not improve top-1 accuracy (stayed at 46.50%), Laya's calibration fitting reduced Expected Calibration Error (ECE) from 0.1533 to 0.1095 and multi-class Brier score from 0.7050 to 0.6899.
5. **Latency**: Local models execute in <60 ms (Laya on Apple M2 MPS) and 0.04 ms (TF-IDF on CPU). Einstein latency is unmeasured as predictions were generated via batch processing inside the Salesforce Sandbox.
