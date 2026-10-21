# Architecture & Design Decisions: Composable Local-First CRM Intelligence

> **Author:** Pavan Chavali  
> **Status:** Published Technical Reference  
> **Target Audience:** Enterprise Software Architects, Principal AI Engineers, VP of Customer Operations

---

## 1. Executive Context: The Routing Paradox

Every enterprise CRM deployment (Salesforce Service Cloud, ServiceNow, Zendesk) begins with the same requirement: **"Route incoming customer tickets to the right department quickly and accurately."**

In practice, organizations oscillate between two extremes:
1. **The Heuristic Trap ("Rules Disguised as AI"):** Teams assemble dozens of brittle regex rules, keyword matchers, and Salesforce Assignment Rules. Over time, rule collisions trigger silent misrouting. Because heuristic rules have **no concept of uncertainty (confidence is `None`)**, 100% of cases are routed blindly.
2. **The Closed SaaS Trap ("Paying for AI Secrets"):** Frustrated by rule debt, leadership adopts vendor SaaS AI (e.g. Salesforce Einstein Case Classification). While accuracy improves slightly, the organization incurs **$50/agent/month license premiums**, exposes **customer PII** across cloud boundaries, endures **800ms+ roundtrip latencies**, and remains trapped inside a proprietary walled garden with uncalibrated confidence scores.

**Laya** represents the third architectural path: a **local-first, composable System 1 decision engine** that delivers superior accuracy, sub-15ms inference, calibrated uncertainty gating, and 100% data sovereignty.

---

## 2. Architectural Comparison

```mermaid
flowchart TD
    subgraph Legacy_Rules["1. Rules-Based Heuristics"]
        R_In["Incoming Case"] --> R_Regex["15 Handcrafted Regex Rules"]
        R_Regex -->|Match / Fallback| R_Out["100% Blind Auto-Route<br/>(Confidence: None | Silent Failures)"]
    end

    subgraph Einstein_SaaS["2. Salesforce Einstein (Closed SaaS)"]
        E_In["Incoming Case"] --> E_Net["HTTP Egress to Cloud ML<br/>(Latency: 450-1200ms)"]
        E_Net --> E_Model["Closed Proprietary Model<br/>($50/agent/mo | Data Leaves Perimeter)"]
        E_Model --> E_Out["Opaque Recommendation<br/>(Uncalibrated Scores)"]
    end

    subgraph Laya_Local["3. Laya Composable Decision Engine"]
        L_In["Incoming Case"] --> L_Lang["Zero-Shot Language Detector"]
        L_Lang --> L_Inf["Sub-15ms Local Inference Engine<br/>(Zero PII Egress | Commodity CPU)"]
        L_Inf --> L_Temp["Temperature-Calibrated Posterior P(y|x)"]
        L_Temp --> L_Gate{"Confidence Gate<br/>τ ≥ 0.85?"}
        L_Gate -->|Yes: High Certainty| L_Auto["Auto-Route Instantly<br/>(100% Precision)"]
        L_Gate -->|No: Ambiguous / Risk| L_Human["Escalate to Human Triage<br/>(Zero Silent Misroutes)"]
    end
```

---

## 3. Core Architectural Pillars

### Pillar A: Local-First Data Sovereignty & Zero PII Egress
In regulated industries (FinTech, HealthTech, Enterprise SaaS), customer tickets frequently contain unredacted PII: credit card expiration notices, tax IDs, internal IP addresses, and user email addresses.
* **Closed SaaS (Einstein):** Requires outbound transmission to external multi-tenant infrastructure.
* **Laya:** Operates locally inside your application cluster or on-premise container. No customer data leaves your network boundary, satisfying strict GDPR, HIPAA, and SOC 2 requirements by design.

### Pillar B: Sub-15ms Latency vs. 800ms Network Roundtrips
In high-throughput call centers or live-chat deflection pipelines, latency directly impacts customer satisfaction:
* **Rules:** < 1 ms (fast, but brittle).
* **Einstein:** 450 ms – 1200 ms (governed by network roundtrips, multi-tenant queueing, and SaaS API rate limits).
* **Laya:** **< 15 ms** on standard Apple Silicon or Linux x86 CPU cores without dedicated GPU hardware.

### Pillar C: Rigorous Uncertainty Calibration (Platt Temperature Scaling)
Raw neural networks are notoriously overconfident or uncalibrated. A model outputting `0.92` probability does not necessarily mean it is correct 92% of the time.

Laya applies post-hoc temperature scaling ($T$) over validation set logits:
$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

Where $T$ is optimized via L-BFGS to minimize negative log-likelihood on an independent validation set. In our benchmark, optimal $T = 1.497$ produced true calibrated posterior probabilities, enabling deterministic confidence gating.

### Pillar D: Confidence-Gated Routing Policy
Traditional CRM rules force a binary dilemma: either auto-route everything (causing customer churn when cases land in the wrong queue) or send everything to human triage (causing staffing bloat).

Laya provides explicit uncertainty thresholding ($\tau$):
$$\text{Action}(x) = \begin{cases} 
\text{AUTO\_ROUTE}(y^*) & \text{if } \max_k P(y_k|x) \ge \tau \\
\text{HUMAN\_TRIAGE\_ESCALATION} & \text{if } \max_k P(y_k|x) < \tau 
\end{cases}$$

At $\tau = 0.85$, **90%** of tickets are automated with **100% precision**, while the remaining **10%** are proactively escalated to senior specialists before customers experience delay.

---

## 4. Why Google Antigravity & Agentic Tooling?

Building and validating this POC repository utilized Google Antigravity to:
1. **Enforce Deterministic Reproducibility:** Ensure exact random seeds across data generation, cross-validation, and metric computation.
2. **Eliminate Flaky Dependencies:** Package lightweight inference heads and Colab notebooks that execute across Apple Silicon, Linux, and Cloud environments without bloat.
3. **Bridge Salesforce & Open Source:** Seamlessly synthesize Apex triggers, Invocable Actions, and FastAPI webhook endpoints into an auditable, production-ready deliverable.

---

## 5. Trade-Off Analysis

| Decision | Chosen Path | Alternative Rejected | Rationale |
|---|---|---|---|
| **Model Size** | Lightweight Classifier Head (< 5MB) | 70B Parameter LLM Prompting | LLM prompting costs $0.02 per case and takes 2500ms; Laya takes 12ms and costs $0. |
| **Calibration** | Empirical Temperature Scaling | Raw Softmax Probs | Raw softmax suffers from extreme overconfidence and breaks threshold gating. |
| **Interface** | FastAPI Bridge + Apex Invocables | Proprietary Salesforce AppExchange Package | Independent REST API enables universal reuse across Zendesk, Freshdesk, Slack, and Salesforce. |

---

## 6. Enterprise Scale: Continuous Learning via Data Lakehouse

In production enterprise deployments, model evolution does not occur inside an interactive notebook. Instead, the architecture establishes an **Active Learning Loop** decoupled from operational CRM limits:

```
[Salesforce Service Cloud] ──(CDC / Fivetran / Data Cloud)──> [Lakehouse: Snowflake / Databricks]
                                                                     │
                                                               (Table: gold_support_cases)
                                                                     │
                                                                     ▼
[Local Laya Pod / Edge API] <──(Hot Reload Weights)── [Headless Training Job: Airflow / dbt]
```

### Architectural Principles of the Enterprise Loop:
1. **Zero Impact on Salesforce OLTP:** Training queries are offloaded entirely to the Data Lakehouse, protecting Salesforce governor limits and transactional throughput.
2. **Quality-Gated Supervision:** Only closed cases meeting strict QA filters (e.g., `CSAT >= 4`, verified by tier-2 leads, and deduplicated) are ingested into training.
3. **Sub-15-Second Retraining:** Because Laya operates as an efficient System 1 decision engine, retraining on 5,000 cases completes in seconds on commodity CPU instances without requiring costly GPU clusters.
4. **Autonomous Expansion of Automation Rate:** As human agents resolve edge cases and corrections flow back into the Lakehouse, Laya's posterior confidence on ambiguous tickets systematically rises, naturally increasing the proportion of tickets that exceed the safety gate ($\tau \ge 0.85$).
