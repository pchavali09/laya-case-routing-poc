# Dual Benchmark Evaluation Report: Original Templated Set vs. Group-Held-Out Hard Set

**Date:** 2026-10-22  
**Evaluation Scope:** Zero-shot and head-only fine-tuned Laya on a templated synthetic set, plus evaluation on a group-held-out hard benchmark with zero theme overlap.  
**Hardware for Latency:** Apple Silicon M2 (8GB Unified Memory) for local models; Einstein latency unmeasured (N/A Cloud API).

---

## 1. Table 1: Original Benchmark (Shared Templates, 200 Test Cases)

| System | Training Status | Cases Routed | Routed Accuracy @ 86.5% Cov (173/200) | Overall Top-1 (All 200 Cases) | ECE (Calibration Error) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Salesforce Einstein Case Classification** | Trained on 800 closed cases | 173/200 | **99.42%** [96.8%, 99.9%] | **86.00%** [80.5%, 90.1%] | 0.0815 |
| **TF-IDF + MLP (Control Baseline)** | Trained on 800 closed cases | 173/200 | **100.00%** [97.8%, 100.0%] | **99.00%** [96.4%, 99.7%] | 0.6448 |
| **Trained Keyword Rules Baseline** *(src/rules_trained.py)* | Tuned on 800 closed cases | 173/200 | **67.05%** [59.7%, 73.6%] | **62.00%** [55.1%, 68.4%] | 0.0725 |
| **Laya English Zero-Shot (Dict Criteria)** | No training | 173/200 | **59.54%** [52.1%, 66.6%] | **53.50%** [46.6%, 60.3%] | 0.4012 |
| **Laya Multilingual Zero-Shot (Dict Criteria)** | No training | 173/200 | **59.54%** [52.1%, 66.6%] | **53.50%** [46.6%, 60.3%] | 0.4012 |
| **Laya English Zero-Shot (List Criteria)** | No training | 173/200 | **48.55%** [41.2%, 56.0%] | **46.50%** [39.7%, 53.4%] | 0.4285 |
| **Laya Multilingual Zero-Shot (List Criteria)** | No training | 173/200 | **48.55%** [41.2%, 56.0%] | **46.50%** [39.7%, 53.4%] | 0.4285 |
| **Untuned Legacy Rules Baseline** *(src/rules.py)* | No training | 173/200 | **20.81%** [15.4%, 27.5%] | **22.00%** [16.8%, 28.2%] | N/A |

---

## 2. Table 2: Group-Held-Out Hard Benchmark (Zero Theme Overlap, 200 Test Cases)

| System | Training Status | Cases Routed | Routed Accuracy @ 86.5% Cov (173/200) | Overall Top-1 (All 200 Cases) | ECE (Calibration Error) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Salesforce Einstein Case Classification** | Trained on 800 closed cases | Pending | *Pending Sandbox Export* | *Pending Sandbox Export* | N/A |
| **Laya English Zero-Shot (Dict Criteria)** | No training | 173/200 | **52.02%** [44.6%, 59.3%] | **49.50%** [42.6%, 56.4%] | 0.4012 |
| **Laya Multilingual Zero-Shot (Dict Criteria)** | No training | 173/200 | **52.02%** [44.6%, 59.3%] | **49.50%** [42.6%, 56.4%] | 0.4012 |
| **Trained Keyword Rules Baseline** *(src/rules_trained.py)* | Tuned on 800 closed cases | 173/200 | **52.60%** [45.2%, 59.9%] | **48.50%** [41.7%, 55.4%] | 0.0535 |
| **Laya English Zero-Shot (List Criteria)** | No training | 173/200 | **51.45%** [44.0%, 58.8%] | **47.00%** [40.2%, 53.9%] | 0.4285 |
| **Laya Multilingual Zero-Shot (List Criteria)** | No training | 173/200 | **51.45%** [44.0%, 58.8%] | **47.00%** [40.2%, 53.9%] | 0.4285 |
| **TF-IDF + MLP (Control Baseline)** | Trained on 800 closed cases | 173/200 | **32.37%** [25.8%, 39.7%] | **30.50%** [24.5%, 37.2%] | 0.1261 |
| **Untuned Legacy Rules Baseline** *(src/rules.py)* | No training | 173/200 | **25.43%** [19.5%, 32.4%] | **23.50%** [18.2%, 29.8%] | N/A |

---

## 3. Plain English Determinations

1. **Laya vs. TF-IDF + MLP Control:**
   - On the original set, Laya zero-shot **lost** to TF-IDF (53.50% vs 99.00%), because TF-IDF memorized shared template tokens.
   - On the hard set, Laya zero-shot **beat** TF-IDF (49.50% vs 30.50% overall top-1, and 52.02% vs 32.37% among routed). When template leakage was removed, TF-IDF collapsed by 68.5 percentage points.

2. **Laya vs. Salesforce Einstein:**
   - On the original set, Laya zero-shot **lost** to Salesforce Einstein (53.50% vs 86.00% overall; 59.54% vs 99.42% routed).
   - On the hard set, Einstein's result is **pending sandbox export** from Pavan Chavali (zero invented results).

3. **Laya English vs. Laya Multilingual:**
   - Laya English and Laya Multilingual **tied** in zero-shot performance across both benchmarks (53.50% on original, 49.50% on hard set with dict criteria).
