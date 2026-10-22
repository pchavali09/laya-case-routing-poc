# Comprehensive Case Routing Architecture Comparison

| Dimension | 1. Rules-Based Heuristics | 2. Salesforce Einstein (Closed SaaS) | 3. Laya (Local-First Calibrated) |
|---|---|---|---|
| **Paradigm** | Handcrafted Regex / If-Else | Cloud Black-Box ML | Local-First System 1 Neural |
| **English Accuracy** | 80.0% (Brittle) | 73.3% | **100.0%** |
| **Spanish Accuracy** | 40.0% (Degraded) | 53.3% | **73.3%** |
| **Hindi Accuracy** | 46.7% (Broken) | 36.7% | **70.0%** |
| **Confidence Scoring** | ❌ None (Always routes 100%) | ⚠️ Relative Score (Uncalibrated) | ✅ **Mathematically Calibrated (T-scaled)** |
| **Uncertainty Gating** | ❌ Impossible | ⚠️ Hard rule cutoffs | ✅ **Threshold Policy (e.g. $\tau=0.85$)** |
| **Inference Latency** | < 1 ms | 450 ms – 1200 ms (API Roundtrip) | **< 15 ms (Local CPU/Worker)** |
| **Data Sovereignty** | Local | ❌ Customer PII leaves perimeter | ✅ **100% Local / Zero egress** |
| **Cost Model** | $0 (High maintenance overhead) | ~$50 / agent / month | **~$20 / month (or $0 on existing VM)** |
| **Composability** | Hardcoded in CRM rules engine | Proprietary Salesforce lock-in | **Pluggable (FastAPI, Python, Edge, Apex)** |
| **Ownership Score** | Low (Maintenance nightmare) | Low (Vendor locked) | **High (100% Owned & Auditable)** |

---

## Confidence-Gating Operational Simulation

How Laya eliminates silent misrouting compared to blind rule engines:

|   threshold |   pct_auto_routed |   pct_human_triage |   auto_routed_accuracy |   auto_routed_error_rate |   cases_auto_routed |   cases_in_triage |
|------------:|------------------:|-------------------:|-----------------------:|-------------------------:|--------------------:|------------------:|
|        0.6  |             100   |                0   |                    100 |                        0 |                  30 |                 0 |
|        0.7  |              93.3 |                6.7 |                    100 |                        0 |                  28 |                 2 |
|        0.8  |              90   |               10   |                    100 |                        0 |                  27 |                 3 |
|        0.85 |              90   |               10   |                    100 |                        0 |                  27 |                 3 |
|        0.9  |              90   |               10   |                    100 |                        0 |                  27 |                 3 |

> **Core Architect Takeaway:**  
> Legacy rules auto-route **100%** of cases with an unavoidable **16.7% error rate** (silent failures that erode customer trust).  
> In contrast, Laya with **$\tau=0.85$** auto-routes high-certainty cases with **near 100% precision**, cleanly flagging ambiguous or high-risk cases for human specialist review.
