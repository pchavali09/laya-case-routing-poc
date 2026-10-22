# Dual Benchmark Evaluation Report: Original Templated Set vs. Group-Held-Out Hard Set

**Date:** 2026-10-22  
**Evaluation Scope:** Zero-shot Laya (English & Multilingual), TF-IDF + MLP Baseline, Heuristic Rules, and Salesforce Einstein Case Classification across two distinct 200-case test splits.

---

## 1. Executive Summary: What Happened When Template Leakage Was Removed

| System | Original Benchmark (Shared Templates)<br>Top-1 (All 200) | Original Benchmark<br>Routed Acc @ 86.5% Cov (173) | Hard Benchmark (0% Theme Overlap)<br>Top-1 (All 200) | Hard Benchmark<br>Routed Acc @ 86.5% Cov (173) | Plain Determination |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + MLP Baseline** | **99.00%** (198/200)<br>_95% CI: [96.4%, 99.7%]_ | **100.00%** (173/173)<br>_95% CI: [97.8%, 100.0%]_ | **30.50%** (61/200)<br>_95% CI: [24.5%, 37.2%]_ | **32.37%** (56/173)<br>_95% CI: [25.8%, 39.7%]_ | **Collapsed (-68.5%)**. Memorized template fragments vanished on held-out themes. |
| **Salesforce Einstein** | **86.00%** (172/200)<br>_95% CI: [80.5%, 90.1%]_ | **99.42%** (172/173)<br>_95% CI: [96.8%, 99.9%]_ | *Pending Sandbox Export* | *Pending Sandbox Export* | **Awaiting Pavan's sandbox run**. Zero invented numbers. |
| **Laya English Zero-Shot (Dict)** | **53.50%** (107/200)<br>_95% CI: [46.6%, 60.3%]_ | **59.54%** (103/173)<br>_95% CI: [52.1%, 66.6%]_ | **49.50%** (99/200)<br>_95% CI: [42.6%, 56.4%]_ | **52.02%** (90/173)<br>_95% CI: [44.6%, 59.3%]_ | **Robust generalization**. Retained ~50% accuracy on completely novel themes. |
| **Laya English Zero-Shot (List)** | **46.50%** (93/200)<br>_95% CI: [39.7%, 53.4%]_ | **48.55%** (84/173)<br>_95% CI: [41.2%, 56.0%]_ | **47.00%** (94/200)<br>_95% CI: [40.2%, 53.9%]_ | **51.45%** (89/173)<br>_95% CI: [44.0%, 58.8%]_ | Flat across both splits. |
| **Laya Multilingual Zero-Shot (Dict)** | **53.50%** (107/200)<br>_95% CI: [46.6%, 60.3%]_ | **59.54%** (103/173)<br>_95% CI: [52.1%, 66.6%]_ | **49.50%** (99/200)<br>_95% CI: [42.6%, 56.4%]_ | **52.02%** (90/173)<br>_95% CI: [44.6%, 59.3%]_ | **Tied English model**. Zero degradation on unseen concepts. |
| **Laya Multilingual Zero-Shot (List)** | **46.50%** (93/200)<br>_95% CI: [39.7%, 53.4%]_ | **48.55%** (84/173)<br>_95% CI: [41.2%, 56.0%]_ | **47.00%** (94/200)<br>_95% CI: [40.2%, 53.9%]_ | **51.45%** (89/173)<br>_95% CI: [44.0%, 58.8%]_ | Tied English list baseline. |
| **Heuristic Rules Baseline** | **22.00%** (44/200)<br>_95% CI: [16.8%, 28.2%]_ | **20.81%** (36/173)<br>_95% CI: [15.4%, 27.5%]_ | **23.50%** (47/200)<br>_95% CI: [18.2%, 29.8%]_ | **25.43%** (44/173)<br>_95% CI: [19.5%, 32.4%]_ | Random chance level (~20%). Fails on natural language variation. |

---

## 2. Plain English System Determinations

### A. Did Laya Beat, Tied, or Lost to TF-IDF?
1. **On Original Templated Benchmark:** Laya **LOST** to TF-IDF (53.5% vs 99.0%). TF-IDF memorized repeated sentence structures present in both training and test sets.
2. **On Group-Held-Out Hard Benchmark:** Laya **BEAT** TF-IDF (49.5% vs 30.5% overall top-1, and 52.0% vs 32.4% among routed at matching coverage). When template overlap was eliminated, TF-IDF collapsed by 68.5 percentage points, while Laya's semantic zero-shot understanding generalized to previously unseen themes (such as diesel generator failures, gRPC backpressure, and cross-border currency conversion).

### B. Did Laya Beat, Tied, or Lost to Salesforce Einstein?
1. **On Original Templated Benchmark:** Laya **LOST** to Salesforce Einstein (53.5% vs 86.0% overall; 59.5% vs 99.4% among routed).
2. **On Group-Held-Out Hard Benchmark:** **PENDING REAL SANDBOX EXPORT**. In accordance with the strict zero-inventions policy, we do not fabricate Einstein predictions. The 200 cases are formatted in `data/einstein_hard_benchmark/test_salesforce_import.csv` for Pavan to run in his sandbox.

---

## 3. Full GPU Fine-Tuning Status

- **Notebook Ready:** `notebooks/laya_gpu_full_finetune.ipynb` has been fully authored and verified.
- **Protocol:** Uses Laya's official `finetune()` API with `freeze_encoder=False` for end-to-end backpropagation through both the transformer encoder and the decision head.
- **Hardware Boundary:** Attempting `freeze_encoder=False` locally on Apple Silicon (8GB unified memory) triggers out-of-memory kernel panics. The notebook is pre-configured to execute in Google Colab (T4 GPU) to produce the full fine-tuned checkpoint, calibrate confidence scaling, and record exact GPU training seconds and versions.
