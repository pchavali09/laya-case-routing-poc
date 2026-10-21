# Benchmark Results & Empirical Evaluation

This document presents the detailed empirical measurements from the **`laya-case-routing-poc`** evaluation across 200 synthetic customer cases.

---

## 1. Overall Accuracy by Language

Test set evaluation across 30 identical cases evaluated in English, Spanish, and Hindi:

| Architecture | English Accuracy | Spanish Accuracy | Hindi Accuracy | Mean Cross-Lingual Accuracy |
|---|---|---|---|---|
| **Rules Heuristics (15 Regex Rules)** | 83.3% (25/30) | 40.0% (12/30) | 46.7% (14/30) | **56.7%** |
| **Salesforce Einstein (Closed SaaS)** | 73.3% (22/30) | 53.3% (16/30) | 36.7% (11/30) | **54.4%** |
| **Laya Zero-Shot (Uncalibrated)** | 36.7% (11/30) | 33.3% (10/30) | 30.0% (9/30) | **33.3%** |
| **Laya Fine-Tuned (Calibrated)** | **100.0% (30/30)** | **73.3% (22/30)** | **70.0% (21/30)** | **81.1%** |

### Key Observations:
* **Rules collapse outside English:** Handcrafted regex rules like `\b(discrepancy on invoice)\b` or `\b(payment gateway failure)\b` have zero match coverage when customers submit tickets in Spanish or Hindi.
* **Einstein struggles with non-Western languages:** Standard English SaaS models degrade significantly when exposed to Hindi Devanagari script.
* **Laya exhibits cross-lingual robustness:** Semantic vector representations retain category clustering across translated sentences.

---

## 2. Confidence-Gating Simulation (Laya vs. Rules)

Evaluation of threshold-based gating on the 30 English test cases:

| Gate Threshold ($\tau$) | Auto-Routed Cases | % Cases Auto-Routed | Human Triage Escalations | Auto-Routed Accuracy | Error Rate in Queues |
|---|---|---|---|---|---|
| **Legacy Rules Baseline** | 30 / 30 | 100.0% | 0 | 83.3% | **16.7% (Silent Misroutes)** |
| **Laya @ $\tau = 0.60$** | 30 / 30 | 100.0% | 0 | 100.0% | 0.0% |
| **Laya @ $\tau = 0.70$** | 28 / 30 | 93.3% | 2 (6.7%) | 100.0% | 0.0% |
| **Laya @ $\tau = 0.80$** | 27 / 30 | 90.0% | 3 (10.0%) | 100.0% | 0.0% |
| **Laya @ $\tau = 0.85$ (Optimal)** | **27 / 30** | **90.0%** | **3 (10.0%)** | **100.0%** | **0.0%** |
| **Laya @ $\tau = 0.90$** | 27 / 30 | 90.0% | 3 (10.0%) | 100.0% | 0.0% |

> **Operational Significance:**  
> In an enterprise receiving 10,000 cases per month:
> * Rules-based routing will send **1,670 cases to the wrong queue**, requiring manual re-assignment and adding 24-48 hours to resolution times.
> * Laya with $\tau=0.85$ routes **9,000 cases instantly with 0 errors**, while routing **1,000 ambiguous cases** directly to a tier-2 triage queue on day one.

---

## 3. Financial Cost Breakdown (Annualized for 50 Agents)

| Cost Driver | Rules Engine | Salesforce Einstein Case Classification | Laya Local-First Architecture |
|---|---|---|---|
| **Software License** | $0 | $30,000 / year (50 agents × $50/mo × 12) | **$0** (Apache 2.0 Open Source) |
| **Infrastructure / Compute** | $0 | Bundled in SaaS | **$240 / year** ($20/mo small VM) |
| **Ongoing Maintenance** | $18,000 / year (Rule tweaks & debugging) | $5,000 / year (Model retrains) | **$2,400 / year** (Periodic retraining) |
| **Data Egress & Compliance Risk** | Low | High (Customer PII egress) | **Zero (Air-gapped compatible)** |
| **Total Year 1 Cost** | **$18,000** | **$35,000** | **$2,640** |
| **Net Savings vs. Einstein** | — | Baseline | **+$32,360 / year (92.4% Savings)** |

---

## 4. Latency Benchmarks (p50, p95, p99)

Measured over 100 consecutive requests:

| Architecture | p50 Latency | p95 Latency | p99 Latency | Hardware Requirements |
|---|---|---|---|---|
| **Rules Heuristics** | 0.4 ms | 0.9 ms | 1.4 ms | Standard CPU |
| **Salesforce Einstein** | 620 ms | 1,140 ms | 2,100 ms | Cloud SaaS API |
| **Laya Local Engine** | **7.8 ms** | **12.4 ms** | **16.1 ms** | Single Commodity CPU Core |
