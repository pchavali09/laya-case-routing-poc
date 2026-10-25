# Measured Benchmark Comparison: Real Laya vs. Baselines vs. Salesforce Einstein

> **Scope of Evaluation:** Zero-shot and head-only fine-tuned Laya evaluated on a templated synthetic dataset of 1,000 cases (800 train, 200 blind test).

| System | Architecture / Artifact | Overall Top-1 Acc (200 Cases) | Wilson 95% CI (Overall) | Acc @ 173 Cases (Matching Einstein) | Wilson 95% CI (Routed) | Measured Latency | Execution Context |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Salesforce Einstein** | Production Einstein Sandbox Model | 86.0% (172/200) | [80.5%, 90.1%] | **99.42%** (172/173) | [96.8%, 99.9%] | Unmeasured (batch in SF) | Cloud SaaS API (27 unrouted) |
| **Simple Local Baseline** | TF-IDF (5k n-grams) + 2-layer MLP | **98.50%** (197/200) | [95.7%, 99.5%] | **100.00%** (173/173) | [97.8%, 100.0%] | **0.04 ms** | Local CPU (synthetic n-gram overlap) |
| **Real Laya Zero-Shot (Dict)** | `convaiinnovations/laya` (Descriptive) | 56.00% (112/200) | [49.1%, 62.7%] | 63.01% (109/173) | [55.6%, 69.8%] | 58.1 ms | Local Apple M2 MPS |
| **Real Laya Zero-Shot (List)** | `convaiinnovations/laya` (Labels only) | 46.50% (93/200) | [39.7%, 53.4%] | 48.55% (84/173) | [41.2%, 56.0%] | 58.0 ms | Local Apple M2 MPS |
| **Real Laya Fine-Tuned** | `convaiinnovations/laya` (Head-only RLCD) | 46.50% (93/200) | [39.7%, 53.4%] | 49.71% (86/173) | [42.3%, 57.1%] | 58.0 ms | Local Apple M2 MPS (frozen encoder) |
| **Heuristic Rules Baseline** | Regex keyword rules (`src/rules.py`) | 22.00% (44/200) | [16.8%, 28.2%] | 20.81% (36/173) | [15.4%, 27.5%] | **0.073 ms** | Local CPU |

### Key Architectural Findings
1. **Einstein Coverage**: Salesforce Einstein correctly abstained on 27 uncertain cases, reaching 99.42% accuracy among routed tickets.
2. **TF-IDF Performance**: The 98.50% top-1 accuracy for TF-IDF is a direct consequence of shared synthetic templates between train and test sets.
3. **Laya Model Status**: On this templated dataset, zero-shot and head-only fine-tuned Laya lost to both Einstein and TF-IDF. Head-only fine-tuning improved probability calibration (ECE reduced from 0.1533 to 0.1095) without altering top-1 accuracy.
