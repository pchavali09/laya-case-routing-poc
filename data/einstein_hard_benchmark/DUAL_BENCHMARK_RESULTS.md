# Dual Benchmark Results: Original vs. Group-Held-Out Hard Benchmark

**Audit Date:** 2026-10-23  
**Branch:** `review/next-round`  
**Laya Version:** Pinned to `laya==0.4.13` (matching local environment and Colab notebook)

---

## 1. Empirical Results: Train-Set by Test-Set Grid (for Trained Systems)

For every trained system, performance is presented across the 2x2 grid of training and test sets.
- **Original 800 Training Set:** `data/einstein_benchmark/train_closed.csv` (`ENS-TRAIN-0001` through `ENS-TRAIN-0800`).
- **Hard 800 Training Set:** `data/einstein_hard_benchmark/train_closed.csv` (`HARD-TRAIN-0001` through `HARD-TRAIN-0800`).
- **Original 200 Test Set:** `data/einstein_benchmark/test_blind.csv` (200 cases).
- **Hard 200 Test Set:** `data/einstein_hard_benchmark/test_blind.csv` (200 cases, 10 group-held-out themes).

All values are empirically measured from repository code and committed prediction artifacts. Unmeasured / pending cells remain unfilled.

| Trained System | Training Set | Test: Original 200 (Overall Top-1) | Test: Original 200 (Routed @173) | Test: Hard 200 (Overall Top-1) | Test: Hard 200 (Routed @173) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Salesforce Einstein Case Classification** | **Original 800** | **86.00%** (172/200)<br><sub>[80.5%, 90.1%]</sub> | **99.42%** (172/173)<br><sub>[96.8%, 99.9%]</sub> | — | — |
| | **Hard 800** | — | — | **trained on hard 800 (pending)** | **trained on hard 800 (pending)** |
| **Trained Keyword Rules Baseline**<br><sub>(Frozen commit `ff8b315`)</sub> | **Original 800** | **73.00%** (146/200)<br><sub>[66.5%, 78.6%]</sub> | **79.19%** (137/173)<br><sub>[72.5%, 84.6%]</sub> | **40.50%** (81/200)<br><sub>[33.9%, 47.4%]</sub> | **45.09%** (78/173)<br><sub>[37.8%, 52.6%]</sub> |
| | **Hard 800** | **55.50%** (111/200)<br><sub>[48.6%, 62.2%]</sub> | **60.12%** (104/173)<br><sub>[52.7%, 67.1%]</sub> | **41.00%** (82/200)<br><sub>[34.4%, 47.9%]</sub> | **43.93%** (76/173)<br><sub>[36.8%, 51.4%]</sub> |
| **TF-IDF + MLP Baseline** | **Original 800** | **99.00%** (198/200)<br><sub>[96.4%, 99.7%]</sub> | **100.00%** (173/173)<br><sub>[97.8%, 100.0%]</sub> | **96.00%** (192/200)<br><sub>[92.3%, 98.0%]</sub> | **100.00%** (173/173)<br><sub>[97.8%, 100.0%]</sub> |
| | **Hard 800** | **42.00%** (84/200)<br><sub>[35.4%, 48.9%]</sub> | **43.35%** (75/173)<br><sub>[36.2%, 50.8%]</sub> | **30.50%** (61/200)<br><sub>[24.5%, 37.3%]</sub> | **32.37%** (56/173)<br><sub>[25.8%, 39.7%]</sub> |
| **Laya Full GPU Fine-Tuned (ModernBERT-large)** | **Hard 800** | pending Colab run | pending Colab run | pending Colab run | pending Colab run |

---

## 2. Reference Table: Untrained & Zero-Shot Baselines (No Training Set)

The following systems undergo zero training. Both test sets are evaluated out-of-the-box.

| Zero-Shot System | Criteria Schema | Test: Original 200 (Overall Top-1) | Test: Original 200 (Routed @173) | Test: Hard 200 (Overall Top-1) | Test: Hard 200 (Routed @173) | Artifact |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Untuned Legacy Rules** | Regex heuristics | **22.00%** (44/200)<br><sub>[16.8%, 28.2%]</sub> | **20.81%** (36/173)<br><sub>[15.4%, 27.5%]</sub> | **23.50%** (47/200)<br><sub>[18.2%, 29.8%]</sub> | **23.70%** (41/173)<br><sub>[17.9%, 30.7%]</sub> | `src/rules.py` |
| **Laya English Zero-Shot** | Dict (clean training-derived) | **56.00%** (112/200)<br><sub>[49.1%, 62.7%]</sub> | **63.01%** (109/173)<br><sub>[55.6%, 69.8%]</sub> | **46.00%** (92/200)<br><sub>[39.2%, 52.9%]</sub> | **53.18%** (92/173)<br><sub>[45.8%, 60.5%]</sub> | `results/predictions_zeroshot_orig_english_dict.csv` |
| **Laya English Zero-Shot** | List (labels only) | **46.50%** (93/200)<br><sub>[39.7%, 53.4%]</sub> | **48.55%** (84/173)<br><sub>[41.2%, 56.0%]</sub> | **47.00%** (94/200)<br><sub>[40.2%, 53.9%]</sub> | **51.45%** (89/173)<br><sub>[44.0%, 58.8%]</sub> | `results/predictions_zeroshot_orig_english_list.csv` |
| **Laya Multilingual Zero-Shot** | Dict (clean training-derived) | **55.50%** (111/200)<br><sub>[48.6%, 62.2%]</sub> | **60.69%** (105/173)<br><sub>[53.3%, 67.7%]</sub> | **54.00%** (108/200)<br><sub>[47.1%, 60.8%]</sub> | **56.07%** (97/173)<br><sub>[48.6%, 63.3%]</sub> | `results/predictions_zeroshot_orig_multilingual_dict.csv` |
| **Laya Multilingual Zero-Shot** | List (labels only) | **45.50%** (91/200)<br><sub>[38.7%, 52.4%]</sub> | **46.82%** (81/173)<br><sub>[39.5%, 54.2%]</sub> | **47.50%** (95/200)<br><sub>[40.7%, 54.4%]</sub> | **52.02%** (90/173)<br><sub>[44.6%, 59.3%]</sub> | `results/predictions_zeroshot_orig_multilingual_list.csv` |

---

## 3. Technical Decisions & Protocol Documentation

### 1. Training Data Provenance (Option A)
- **Which dataset the Einstein sandbox saw:** The Salesforce Einstein sandbox model was trained on the **original 800 closed cases** (`data/einstein_benchmark/train_closed.csv`, `ENS-TRAIN-0001` through `ENS-TRAIN-0800`). It has never seen the hard benchmark cases.
- **Option A Selection:** `data/einstein_hard_benchmark/train_closed.csv` (800 closed cases, `HARD-TRAIN-0001` through `HARD-TRAIN-0800`, balanced across 20 themes with 40 cases per theme) is used everywhere for the Colab notebook and all trained baselines.
- **Salesforce Import File:** Generated `data/einstein_hard_benchmark/train_salesforce_import.csv` with the exact Salesforce schema (`ExternalId__c, Subject, Description, Origin, Priority, Status, Department__c`) so Pavan can import and retrain Einstein in the sandbox if desired.
- **Notebook Step 4 Validation Split:** Step 4 in `notebooks/laya_gpu_full_finetune.ipynb` unpacks and reads `data/einstein_hard_benchmark/train_closed.csv`. `VAL_THEMES` successfully isolates 200 validation cases and 600 training cases strictly by theme.

### 2. Criteria Cleaning & Explanation of 56.0% to 53.5% Shift
- **Clean Criteria Baseline:** In `run_full_benchmark.py`, original dictionary criteria were written as:
  ```python
  DICT_CRITERIA = {
      "Billing": "invoices, payments, refunds, charges, subscriptions, pricing",
      "Technical": "system errors, software bugs, outages, API integration, technical issues",
      "Sales": "quotes, enterprise licensing, contract renewals, upgrades, sales demos",
      "Operations": "shipping, logistics, warehouse, facilities, hardware procurement",
      "Other": "general inquiries, uncategorized corporate communications, miscellaneous feedback"
  }
  ```
  This scored **56.00% (112/200)** on the original test set.
- When creating `run_dual_benchmark.py`, terms like `"credit cards, bank debits, currency fees"`, `"generator power, customs freight"`, `"SDK exceptions, rate limits"`, `"SLAs, competitor buyouts"`, and `"academic grants, sponsorships"` were added to the criteria dictionary. These terms were derived from the hard-set test themes (`currency_conversion_and_cross_border_fees`, `critical_power_and_generator_systems`, etc.).
- **Mechanism of Shift:** Adding vocabulary from the held-out test themes to category descriptions altered the contextual classification boundaries of the zero-shot model. Specifically, 5 borderline cases in the original test set that previously classified correctly were misclassified under the altered descriptions, dropping accuracy to **53.50% (107/200)**.
- **Resolution:** Purging all hard-set wording and restoring descriptions written strictly from training themes eliminated this contamination and restored measured accuracy on the original test set to exactly **56.00% (112/200)**. Labels-only (list criteria) runs remain untouched at **46.50% (93/200)**.

### 3. Overlap Audit Summary
`src/einstein_benchmark/check_theme_overlap.py` audits the 200 hard test cases against **ALL** data any system trains on:
1. **Hard Train (800 cases):** 0 shared themes, 0 text duplicates, 0.52% 3-gram Jaccard, mean max sim = 0.1578.
2. **Original Train (800 cases):** not measurable (no theme labels), 0 text duplicates, 0.48% 3-gram Jaccard, mean max sim = 0.2380.
3. **Combined Train (1,600 cases):** 0 shared themes (against labeled hard train), 0 text duplicates, 0.47% 3-gram Jaccard, mean max sim = 0.2387, max sim = 0.5754 (no near-duplicates). Full report saved to `data/einstein_hard_benchmark/THEME_OVERLAP_REPORT.md`.

### 4. Rewritten Trained Keyword Rules Baseline
- `src/rules_trained.py` was completely rewritten based strictly and exclusively on the 800 training cases in `data/einstein_hard_benchmark/train_closed.csv`.
- Neither test set was ever referenced or inspected.
- The rule code was frozen and committed to git at commit `ff8b315a40e82ea79f78203ac03f17ab6ba58f86` before scoring.
- Measured performance:
  - On Original 200: **55.50%** (111/200) Top-1; **60.12%** (104/173) Routed @173.
  - On Hard 200: **41.00%** (82/200) Top-1; **43.93%** (76/173) Routed @173.

### 5. Checkpoint Reloading API and Colab Notebook Settings
- **Pinned Version:** `laya==0.4.13` (matching the local virtualenv). Version verification uses `importlib.metadata.version("laya")` with no fallback string.
- **Batching Parameters:** `micro_batch=4, grad_accum=16` in `TrainConfig`.
- **GPU Memory Management:** Base model is loaded only to snapshot reference weights for divergence verification, and is deleted immediately via `del base_agent` and `torch.cuda.empty_cache()` before fine-tuning starts.
- **Question Schema Alignment:** Training passes `question_id="department"` and `instructions="Which department should handle this customer support case?"` into `TrainConfig`. Scoring in `evaluate_reloaded_agent` uses the exact matching choice schema (`criteria=DEPARTMENTS`, instructions, and question_id).
- **Text-Level Integrity Assertions:** Asserts zero identical `Subject` or `Description` strings between train/val splits and either test set.
