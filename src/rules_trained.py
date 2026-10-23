"""
Keyword Rules Baseline: Trained on 800 Training Cases (train_closed.csv ONLY)
=============================================================================
Methodology & Strict Data Leakage Prevention:
---------------------------------------------
1. This rule set was authored strictly by analyzing `data/einstein_hard_benchmark/train_closed.csv` (800 cases).
2. Neither test set (`data/einstein_benchmark/test_blind.csv` nor `data/einstein_hard_benchmark/test_blind.csv`)
   was ever inspected, queried, or referenced during rule authoring.
3. Feature engineering based on top discriminative n-grams per department across the 20 training themes:
   - Billing: tax exemption/VAT, invoice line items, gateway/ACH failures, license seat proration
   - Technical: API 500/payload errors, SSO/SAML auth, DB latency/pool exhaustion, container/pod crashes
   - Sales: prepay discounts/volume quotes, product demos/walkthroughs, contract renewals, partner/reseller agreements
   - Operations: facilities/HVAC controls, hardware asset tagging/docking, RFID badges, office moves/desks
   - Other: DMCA/copyright notices, GDPR/CCPA data privacy/erasure, marketing unsubscribes, recruiting/employment
"""

import re
from typing import Tuple

def classify_case_trained_rules(subject: str, body: str) -> Tuple[str, float, str]:
    """
    Classifies a support case based on keywords extracted strictly from
    data/einstein_hard_benchmark/train_closed.csv (800 cases).
    Returns: (predicted_department, confidence, rule_name)
    """
    text = f"{subject} {body}".lower()

    # Tier 1: High-confidence primary discriminative patterns from the 20 training themes
    if re.search(r"\b(invoice|itemized.*receipt|accounting ledger|tax residency|w-9 form|withholding tax|vat compliance|gateway.*failure|ach payment|credit memo|proration|billing statement|direct debit|chargeback|wire transfer|remittance|refund)\b", text):
        return "Billing", 0.90, "RULE_TRAINED_BILLING_PRIMARY"

    if re.search(r"\b(json payload|api gateway|saml assertion|sso.*login|500 internal|endpoint dispatch|connection pool|database.*latency|kubernetes|crash loop backoff|worker deployment|stack trace|oomkilled|fatal error|unhandled exception)\b", text):
        return "Technical", 0.90, "RULE_TRAINED_TECH_PRIMARY"

    if re.search(r"\b(server room hvac|temperature alarm|docking station|rfid.*badge|security badge|thermostat calibration|ergonomic chair|standing desk|desk reconfiguration|office move|asset tagging|cleanroom|pallet|freight delivery)\b", text):
        return "Operations", 0.90, "RULE_TRAINED_OPS_PRIMARY"

    if re.search(r"\b(dmca copyright|takedown notification|ccpa personal|data erasure|gdpr article|press release|executive interview|job applicant|employment verification|unsubscribe|promotional email|trademark infringement)\b", text):
        return "Other", 0.90, "RULE_TRAINED_OTHER_PRIMARY"

    if re.search(r"\b(prepay discount|enterprise licensing|product demo|technical walkthrough|contract renewal|trial account|channel partner|commission agreement|volume quote|pricing schedule|poc evaluation|tier pricing)\b", text):
        return "Sales", 0.90, "RULE_TRAINED_SALES_PRIMARY"

    # Tier 2: Secondary broader vocabulary from train_closed.csv
    if re.search(r"\b(tax|vat|invoice|receipt|payment|charge|billing|ledger)\b", text):
        return "Billing", 0.70, "RULE_TRAINED_BILLING_SECONDARY"

    if re.search(r"\b(api|saml|sso|database|pod|kubernetes|crash|error|latency)\b", text):
        return "Technical", 0.70, "RULE_TRAINED_TECH_SECONDARY"

    if re.search(r"\b(licensing|renewal|demo|quote|pricing|reseller|partner|discount)\b", text):
        return "Sales", 0.70, "RULE_TRAINED_SALES_SECONDARY"

    if re.search(r"\b(hvac|badge|inventory|facilities|thermostat|desk|docking|warehouse)\b", text):
        return "Operations", 0.70, "RULE_TRAINED_OPS_SECONDARY"

    if re.search(r"\b(dmca|copyright|gdpr|ccpa|press|applicant|interview|unsubscribe|takedown|legal)\b", text):
        return "Other", 0.70, "RULE_TRAINED_OTHER_SECONDARY"

    # Tier 3: Low-confidence default fallback
    return "Other", 0.20, "RULE_TRAINED_DEFAULT_FALLBACK"
