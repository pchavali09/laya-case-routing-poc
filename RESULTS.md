# Measured Benchmark Results: Real Laya vs. Baselines vs. Salesforce Einstein

This document records the empirical results measured across the 1,000-case Einstein benchmark:
* **Training set**: 800 historical closed cases (`data/einstein_benchmark/train_closed.csv`)
* **Test set**: 200 blind test cases (`data/einstein_benchmark/test_blind.csv`)
* **Ground truth**: 200 labels with difficulty breakdown (`data/einstein_benchmark/test_truth.csv`)
* **Salesforce Einstein baseline**: Real export from a production Salesforce Sandbox (`data/einstein_benchmark/einstein_predictions.csv`)
* **Hardware**: Apple M2 (MacBook Air), 8GB Unified Memory, PyTorch with MPS backend.
* **Scope**: Evaluates **zero-shot and head-only fine-tuned Laya on a templated synthetic set**.

---

## 1. Overall Head-to-Head Performance (with Wilson 95% Confidence Intervals)

Scored on the exact 200 blind cases. Overall top-1 accuracy evaluates every case without abstention. Matched coverage evaluates accuracy on the 173 cases where the model or Einstein routes.

| System / Run | Model Artifact | Overall Top-1 Acc (200 Cases) | Wilson 95% CI (Overall) | Acc @ 173 Cases (Matching Einstein) | Wilson 95% CI (Routed) | Measured Latency (Mean) | Execution Context |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Salesforce Einstein** | Production Einstein Sandbox Model | 86.0% (172/200 total) | [80.5%, 90.1%] | **99.42%** (172/173 routed) | [96.8%, 99.9%] | Unmeasured (batch in SF) | Cloud SaaS API (27 unrouted) |
| **Simple Local Baseline** | TF-IDF (5k n-grams) + 2-layer MLP (hidden=64) | **98.50%** (197/200) | [95.7%, 99.5%] | **100.00%** (173/173 routed) | [97.8%, 100.0%] | **0.04 ms** | Local CPU (synthetic n-gram overlap) |
| **Real Laya Zero-Shot (Dict)** | `convaiinnovations/laya` (Descriptive Criteria) | 56.00% (112/200) | [49.1%, 62.7%] | 63.01% (109/173 routed) | [55.6%, 69.8%] | 58.1 ms | Local Apple M2 (MPS) |
| **Real Laya Zero-Shot (List)** | `convaiinnovations/laya` (Choice Labels only) | 46.50% (93/200) | [39.7%, 53.4%] | 48.55% (84/173 routed) | [41.2%, 56.0%] | 58.0 ms | Local Apple M2 (MPS) |
| **Real Laya Fine-Tuned** | `convaiinnovations/laya` (Head-only RLCD, ECE=0.1095) | 46.50% (93/200) | [39.7%, 53.4%] | 49.71% (86/173 routed) | [42.3%, 57.1%] | 58.0 ms | Local Apple M2 MPS (frozen encoder) |
| **Heuristic Rules Baseline** | Regex keyword rules (`src/rules.py`) | 22.00% (44/200) | [16.8%, 28.2%] | 20.81% (36/173 routed) | [15.4%, 27.5%] | **0.073 ms** | Local CPU (written for initial POC schema) |

> **Abstention Handling:** In Salesforce Einstein, 27 cases had confidence below the automated routing threshold and were left unrouted for human agent triage. In a like-for-like comparison, these 27 unrouted cases are treated as abstentions, not misclassifications.

---

## 2. Accuracy Breakdown by Difficulty

The 200 test cases are split across four difficulty tiers:
* **Easy** (80 cases, 40%): Clear single-department vocabulary
* **Paraphrase** (60 cases, 30%): Semantic synonyms without exact keywords
* **Multi-Topic** (40 cases, 20%): Mentions multiple departments (e.g. outage + refund request)
* **Ambiguous** (20 cases, 10%): High entropy edge cases

| System / Model | Easy (80 Cases) | Paraphrase (60 Cases) | Multi-Topic (40 Cases) | Ambiguous (20 Cases) | Overall (200 Cases) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Salesforce Einstein (Routed: 173)** | 100.0% (75/75, 5 unrouted) | 100.0% (48/48, 12 unrouted) | 97.1% (33/34, 6 unrouted) | 100.0% (16/16, 4 unrouted) | **99.4% (172/173)** |
| **Simple Local Baseline (TF-IDF + MLP)** | 98.8% (79/80) | 100.0% (60/60) | 95.0% (38/40) | 100.0% (20/20) | **98.5% (197/200)** |
| **Real Laya Zero-Shot (Dict Criteria)** | 65.0% (52/80) | 53.3% (32/60) | 37.5% (15/40) | 65.0% (13/20) | **56.0% (112/200)** |
| **Real Laya Zero-Shot (List Criteria)** | 58.8% (47/80) | 41.7% (25/60) | 30.0% (12/40) | 45.0% (9/20) | **46.5% (93/200)** |
| **Real Laya Fine-Tuned (Head)** | 58.8% (47/80) | 41.7% (25/60) | 30.0% (12/40) | 45.0% (9/20) | **46.5% (93/200)** |
| **Heuristic Rules Baseline** | 23.8% (19/80) | 20.0% (12/60) | 22.5% (9/40) | 20.0% (4/20) | **22.0% (44/200)** |

---

## 3. Threshold Coverage & Confidence-Gating Curves

### Simple Local Baseline (TF-IDF + MLP)
| Cases Routed | Coverage % | Cutoff Confidence | Correct Cases | Accuracy Among Routed | Cases Unrouted |
|---|---|---|---|---|---|
| **200** | 100.0% | 0.2123 | 197 | 98.50% | 0 |
| **190** | 95.0% | 0.2386 | 189 | 99.47% | 10 |
| **173** (Matches Einstein) | 86.5% | 0.2520 | 173 | **100.00%** | 27 |
| **150** | 75.0% | 0.2658 | 150 | 100.00% | 50 |
| **100** | 50.0% | 0.2888 | 100 | 100.00% | 100 |

### Real Laya Zero-Shot (Descriptive Dictionary Criteria)
| Cases Routed | Coverage % | Cutoff Confidence | Correct Cases | Accuracy Among Routed | Cases Unrouted |
|---|---|---|---|---|---|
| **200** | 100.0% | 0.0079 | 112 | 56.00% | 0 |
| **190** | 95.0% | 0.0220 | 111 | 58.42% | 10 |
| **173** (Matches Einstein) | 86.5% | 0.0432 | 109 | **63.01%** | 27 |
| **150** | 75.0% | 0.0758 | 99 | 66.00% | 50 |
| **100** | 50.0% | 0.2225 | 71 | 71.00% | 100 |

### Real Laya Zero-Shot (Choice Label List Criteria)
| Cases Routed | Coverage % | Cutoff Confidence | Correct Cases | Accuracy Among Routed | Cases Unrouted |
|---|---|---|---|---|---|
| **200** | 100.0% | 0.0062 | 93 | 46.50% | 0 |
| **190** | 95.0% | 0.0389 | 91 | 47.89% | 10 |
| **173** (Matches Einstein) | 86.5% | 0.0760 | 84 | **48.55%** | 27 |
| **150** | 75.0% | 0.1383 | 73 | 48.67% | 50 |
| **100** | 50.0% | 0.2779 | 55 | 55.00% | 100 |

### Real Laya Fine-Tuned (Head + Temperature Calibration, RLCD)
| Cases Routed | Coverage % | Cutoff Confidence | Correct Cases | Accuracy Among Routed | Cases Unrouted |
|---|---|---|---|---|---|
| **200** | 100.0% | 0.2199 | 93 | 46.50% | 0 |
| **190** | 95.0% | 0.2445 | 91 | 47.89% | 10 |
| **173** (Matches Einstein) | 86.5% | 0.2600 | 86 | **49.71%** | 27 |
| **150** | 75.0% | 0.3061 | 75 | 50.00% | 50 |
| **100** | 50.0% | 0.3951 | 55 | 55.00% | 100 |

---

## 4. Empirical Conclusions

> **Scoped Statement:** All conclusions below apply strictly to **zero-shot and head-only fine-tuned Laya evaluated on a templated synthetic dataset**:

1. **Outcome vs. Salesforce Einstein**: Real Laya **lost** to Salesforce Einstein. Einstein achieved 99.42% accuracy among routed cases (172/173) in the production sandbox, while Laya reached 48.55% (labels only) to 63.01% (descriptive criteria) at the equivalent 173-case coverage point.
2. **Outcome vs. Simple Local Baseline**: Real Laya **lost** to the classical TF-IDF + MLP baseline. The TF-IDF model achieved 98.50% overall top-1 accuracy (197/200) and 100.0% accuracy on the top 173 cases, with 0.04ms average latency, because train and test sets share template sentence structures.
3. **Laya Model Behavior**: `convaiinnovations/laya` operates as an open-weights general router with 421M parameters. Without domain-specific encoder fine-tuning, zero-shot accuracy is 46.5% with label lists and 56.0% with detailed dictionary descriptions. Fine-tuning the classification head with frozen encoder weights on 800 closed cases improved calibration (ECE dropped from 0.1533 to 0.1095), but did not boost top-1 accuracy.
