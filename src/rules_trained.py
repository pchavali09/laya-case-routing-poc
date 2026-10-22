"""
Keyword Rules Baseline: Tuned from 800 Training Cases (train_closed.csv ONLY)
=============================================================================
Methodology & Data Leakage Prevention:
-------------------------------------
1. This rule set was authored strictly by analyzing `data/einstein_benchmark/train_closed.csv` (800 cases).
2. Neither test set (`data/einstein_benchmark/test_blind.csv` nor `data/einstein_hard_benchmark/test_blind.csv`)
   was ever inspected, queried, or referenced during authoring.
3. Feature engineering based on top discriminative n-grams per department in train_closed.csv:
   - Billing: 'invoice', 'credit', 'payment', 'charge', 'statement', 'dollars', 'card', 'vat', 'drawdown', 'balance'
   - Technical: 'api', 'database', 'replica', 'cluster', 'endpoint', 'error', 'timeout', 'health check', 'crash'
   - Sales: 'enterprise', 'procurement', 'licensing', 'discount', 'contract renewal', 'demo', 'seats', 'quote'
   - Operations: 'warehouse', 'freight', 'shipment', 'facilities', 'badge', 'hvac', 'pallet', 'cleanroom', 'dock'
   - Other: 'dmca', 'copyright', 'privacy', 'gdpr', 'ccpa', 'inquiry', 'community', 'interview', 'panel', 'recruiting'
"""

import re
from typing import Tuple, Optional

def classify_case_trained_rules(subject: str, body: str) -> Tuple[str, float, str]:
    """
    Classifies a case based on keywords extracted from the 800 training cases.
    Returns: (predicted_department, confidence, rule_name)
    """
    text = f"{subject} {body}".lower()

    # Rule 1: High-confidence Billing terms from train_closed.csv
    if re.search(r"\b(invoice|credit memo|payment failure|card declined|vat number|drawdown account|overcharged|wire transfer|remittance|refund|billing statement|pro-rata|proration)\b", text):
        return "Billing", 0.90, "RULE_TRAINED_BILLING_PRIMARY"

    # Rule 2: High-confidence Technical terms from train_closed.csv
    if re.search(r"\b(rest api|replica database|connection pool|500 internal|timeout|crashloopbackoff|docker|kubernetes|oomkilled|health check probe|stack trace|nullpointer|endpoint error)\b", text):
        return "Technical", 0.90, "RULE_TRAINED_TECH_PRIMARY"

    # Rule 3: High-confidence Operations terms from train_closed.csv
    if re.search(r"\b(damaged server chassis|freight delivery|keycard reader|cleanroom access|warehouse|crushed diagnostic|ventilation fan|security pre clearance|hvac|asset tagging|pallet)\b", text):
        return "Operations", 0.90, "RULE_TRAINED_OPS_PRIMARY"

    # Rule 4: High-confidence Other terms from train_closed.csv (legal, privacy, recruiting, PR)
    if re.search(r"\b(gdpr|ccpa|dmca copyright|trademark infringement|personal data portability|erasure request|panel discussion|lost property|moderation inquiry|press release|employment verification)\b", text):
        return "Other", 0.90, "RULE_TRAINED_OTHER_PRIMARY"

    # Rule 5: High-confidence Sales terms from train_closed.csv
    if re.search(r"\b(contract renewal|volume discount|tier pricing|reseller agreement|product demo|trial account|commercial framework|procurement committee|quote for self hosted|enterprise licensing)\b", text):
        return "Sales", 0.90, "RULE_TRAINED_SALES_PRIMARY"

    # Secondary / Broader patterns from train set
    if re.search(r"\b(dollars|banking|clearing house|charges|dispute|debit)\b", text):
        return "Billing", 0.70, "RULE_TRAINED_BILLING_SECONDARY"
    if re.search(r"\b(latency|query|logs|sync lag|bug|unhandled)\b", text):
        return "Technical", 0.70, "RULE_TRAINED_TECH_SECONDARY"
    if re.search(r"\b(procurement|quote|pricing|seats|upgrade|proposal)\b", text):
        return "Sales", 0.70, "RULE_TRAINED_SALES_SECONDARY"
    if re.search(r"\b(facilities|courier|air freight|shipping|logistics|badge)\b", text):
        return "Operations", 0.70, "RULE_TRAINED_OPS_SECONDARY"
    if re.search(r"\b(legal|privacy|charity|donation|conference sponsorship|symposium)\b", text):
        return "Other", 0.70, "RULE_TRAINED_OTHER_SECONDARY"

    # Default fallback
    return "Other", 0.20, "RULE_TRAINED_DEFAULT_FALLBACK"
