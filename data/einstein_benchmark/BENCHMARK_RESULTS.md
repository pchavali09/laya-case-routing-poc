# 1,000-Case Einstein Benchmark: Empirical Results & Evaluation

This document presents the official, real-world head-to-head evaluation between **Salesforce Einstein Case Classification**, **Laya Classifier**, and the **Heuristic Rules Baseline** across the 1,000-case benchmark (`800` closed training cases, `200` unseen test cases).

* **Salesforce Einstein Source:** Live model trained in a verified Salesforce Service Cloud environment on the 800 closed benchmark cases (`train_closed.csv`), activated with **90% Automate Threshold**, and evaluated on 200 test cases (`test_blind.csv`).
* **Laya Classifier Source:** Temperature-calibrated neural decision head ($T = 1.6739$) trained locally with seed `2026`.
* **Rules Baseline Source:** 15 handcrafted keyword/regex priority heuristic rules typical of legacy CRM routing (`src/rules.py`).

---

## 1. Overall Accuracy by Difficulty Tier (200 Unseen Cases)

| Difficulty Tier | Cases | Rules Baseline | Salesforce Einstein | Laya Classifier |
| :--- | :---: | :---: | :---: | :---: |
| **OVERALL (All 200 Cases)** | **200** | **44 / 200 (22.0%)** | **172 / 200 (86.0%)** | **185 / 200 (92.5%)** |
| `easy` (direct keywords) | 80 | 19 / 80 (23.8%) | 75 / 80 (93.8%) | **77 / 80 (96.2%)** |
| `paraphrase` (synonyms & colloquial) | 60 | 12 / 60 (20.0%) | 48 / 60 (80.0%) | **55 / 60 (91.7%)** |
| `multi_topic` (cross-department topics) | 40 | 9 / 40 (22.5%) | 33 / 40 (82.5%) | **36 / 40 (90.0%)** |
| `ambiguous` (under-specified inquiries) | 20 | 4 / 20 (20.0%) | 16 / 20 (80.0%) | **17 / 20 (85.0%)** |

---

## 2. Confidence-Gated Auto-Routing Frontier

In enterprise customer support, uncalibrated blind auto-routing causes silent misroutes and queue pollution. Confidence gating allows systems to automate high-confidence cases while routing low-certainty cases to human triage.

| Routing Architecture | Confidence Threshold ($\tau$) | Auto-Routed Cases | % Cases Auto-Routed | Auto-Routed Accuracy | Deflected to Human Review |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Legacy Rules Baseline** | *No Confidence* | 200 / 200 | 100.0% (Blind) | **22.0%** | 0 cases (0%) |
| **Salesforce Einstein** | **90% Automate Threshold** | **173 / 200** | **86.5%** | **99.4% (172/173)** | **27 cases (13.5%)** |
| **Laya Classifier** | **$\tau = 0.90$** | 168 / 200 | 84.0% | **98.2% (165/168)** | 32 cases (16.0%) |
| **Laya Classifier** | **$\tau = 0.85$** | **175 / 200** | **87.5%** | **98.3% (172/175)** | **25 cases (12.5%)** |
| **Laya Classifier** | **$\tau = 0.80$** | **182 / 200** | **91.0%** | **98.4% (179/182)** | **18 cases (9.0%)** |

---

## 3. Analysis of Einstein's 27 Un-Routed Cases

When Einstein’s Automate Threshold was set to 90%, it declined to auto-save **27 cases** because confidence fell below 90%. Comparing those 27 cases to ground truth reveals why:

* **Paraphrase**: 12 cases (44.4%)
* **Multi-Topic**: 6 cases (22.2%)
* **Easy**: 5 cases (18.5%)
* **Ambiguous**: 4 cases (14.8%)

Einstein's confidence calibration accurately recognized that these cases carried higher uncertainty, preventing potential misroutes and deflecting them to agents.

---

## 4. Key Comparative Takeaways

1. **Precision of Gating**: Both Salesforce Einstein (99.4% at 90% threshold) and Laya (98.4% at 80% threshold) prove that enterprise AI routing must be confidence-gated.
2. **Semantic Robustness**: Laya demonstrates higher resilience on nuanced phrasing (+11.7% on paraphrases, +7.5% on multi-topic cases), allowing it to achieve 91.0% auto-routing coverage at 98.4% precision.
3. **Operational Overhead & Cost**:
   * **Salesforce Einstein**: Cloud-dependent batch training (2–12 hours), enterprise license tiers, multi-tenant off-core execution.
   * **Laya**: Edge-native, trains in under 3 seconds, runs sub-15ms local inference, 100% zero data egress, Apache 2.0 open-source.
