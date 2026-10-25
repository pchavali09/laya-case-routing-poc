# Measured Benchmark Results: Real Laya vs. Baselines vs. Salesforce Einstein

This document records the empirical results measured on the 200-case hard CRM benchmark:
* **Training set**: 800 narrow cases (`data/einstein_hard_benchmark/train_closed.csv`, ~132 distinct descriptions after masking digits).
* **Test set**: 200 blind test cases (`data/einstein_hard_benchmark/test_blind.csv`, 10 novel themes absent from training).
* **Ground truth**: 200 labels with difficulty breakdown (`data/einstein_hard_benchmark/test_truth.csv`).
* **Salesforce Einstein baseline**: Real export from a production Salesforce Sandbox (`data/einstein_hard_benchmark/einstein_hard_raw_export.csv`).
* **Hardware**: Apple M2 (MacBook Air), 8GB Unified Memory for local inference; Google Colab Tesla T4 GPU for full transformer fine-tuning.

---

## 1. Measured Head-to-Head Results (200-Case Hard Test)

Scored on the exact 200 blind cases. Overall top-1 accuracy evaluates every case without abstention.

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

## 2. Multi-Seed Laya Fine-Tuning Breakdown

Full encoder fine-tuning of the 421.3M-parameter ModernBERT-large Laya model (`convaiinnovations/laya`) trained strictly on the 800 narrow cases (`data/einstein_hard_benchmark/train_closed.csv`) and scored on the 200 hard test cases (`data/einstein_hard_benchmark/test_blind.csv`):
Fine-tuning ran on a Tesla T4 GPU (`micro_batch=4, grad_accum=16`, effective batch size 64, AdamW `lr=2.5e-5` encoder / `1e-4` head) across 3 distinct random seeds:

| Seed | Model & Training Data | Eval Set | Overall Top-1 Acc | Wilson 95% CI | ECE | Routed @ conf ≥ 0.90 (Cases, Accuracy) |
| :---: |---|:---:|:---:|:---:|:---:|:---:|
| **2026** | Laya (ModernBERT-large) trained on Narrow 800 | Hard 200 | **97.00%** (194/200) | [93.6%, 98.6%] | 0.0167 | 192/193 = **99.5%** (96.5% coverage) |
| **2027** | Laya (ModernBERT-large) trained on Narrow 800 | Hard 200 | **100.00%** (200/200) | [98.1%, 100.0%] | 0.0119 | 194/194 = **100.00%** (97.0% coverage) |
| **2028** | Laya (ModernBERT-large) trained on Narrow 800 | Hard 200 | **99.00%** (198/200) | [96.4%, 99.7%] | 0.0894 | 124/124 = **100.00%** (62.0% coverage) |
| **Mean** | Laya (ModernBERT-large) trained on Narrow 800 | Hard 200 | **98.67% ± 1.25%** | — | **0.0393** | **99.83%** (85.2% mean coverage) |

Transfer test of these same checkpoints on the earlier test set yielded **45.50% ± 8.13%** (Seed 2026: 51.0%, Seed 2027: 51.5%, Seed 2028: 34.0%), demonstrating that the 98.7% score reflects in-distribution specialization to the 20 training themes.
