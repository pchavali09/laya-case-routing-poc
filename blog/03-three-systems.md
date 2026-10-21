# Benchmarking Heuristics vs. Closed SaaS vs. Laya: An Empirical Study

*By Pavan Chavali · 9 min read*

---

To move beyond marketing slides, we built a fully reproducible public benchmark repository: **`laya-case-routing-poc`**.

We synthesized a rigorous dataset of 200 realistic enterprise support cases across 5 functional departments (Billing, Technical, Sales, Operations, and Other) with varying urgency and churn risk indicators. We then split the data into 140 training cases, 30 validation cases, and 30 test cases, evaluating the test set across **English, Spanish, and Hindi**.

We benchmarked three distinct paradigms:
1. **Rules Baseline:** 15 handcrafted regex heuristics.
2. **Salesforce Einstein:** Modeled SaaS black-box classifier.
3. **Laya:** A local-first, temperature-calibrated System 1 decision engine.

---

## The Raw Numbers

| Architecture | English Accuracy | Spanish Accuracy | Hindi Accuracy | Inference Latency | PII Leaves Perimeter? | Annual License Fee (50 Agents) |
|---|---|---|---|---|---|---|
| **Rules Heuristics** | 83.3% | 40.0% | 46.7% | < 1 ms | ❌ No | $0 |
| **Salesforce Einstein** | 73.3% | 53.3% | 36.7% | 620 ms | ⚠️ Yes | $30,000 / year |
| **Laya Local Engine** | **100.0%** | **73.3%** | **70.0%** | **7.8 ms** | ❌ **No** | **$0** |

---

## Key Finding 1: The Cross-Lingual Cliff
Handcrafted regex rules work reasonably well on standard English phrasing (83.3%). But the moment customers submit tickets in Spanish or Hindi, accuracy plummets to 40% and 46.7%. Because the English keywords (`"invoice"`, `"webhook"`) do not exist in foreign scripts, cases fall through to default fallback queues.

Laya, by contrast, leverages rich semantic representations that retain cross-lingual concept mapping, delivering **73.3% in Spanish** and **70.0% in Hindi** without separate translation pipelines.

---

## Key Finding 2: Confidence-Gated Safety
The true superpower of Laya is **uncertainty estimation**. Through post-hoc Platt temperature scaling on the validation set, Laya fits an optimal temperature parameter ($T = 1.497$), transforming raw neural logits into true calibrated probabilities.

This allows us to implement a **Confidence Gate**:
* If $P \ge 0.85$, auto-route the case instantly.
* If $P < 0.85$, escalate to a human specialist.

In our simulation, setting $\tau = 0.85$ allowed the system to **auto-route 90% of cases with 100% precision**, completely eliminating the silent misrouting errors that plague legacy rules.

In Part 4, we examine how architects can deploy composable, local-first decision engines into their own stacks.
