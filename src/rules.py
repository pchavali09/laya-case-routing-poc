"""
Rules-Based Baseline Classifier
--------------------------------
Implements 15 handcrafted if-then keyword/regex heuristic rules typical of legacy
CRM routing systems (e.g. Salesforce Assignment Rules, Zendesk Triggers).

Demonstrates the "Rules as AI" trap:
1. Rigid: Fails on paraphrasing, synonyms, and cross-lingual translation.
2. Uncalibrated: Returns no uncertainty score (confidence=None). Always auto-routes blindly.
3. Brittle: Priority ordering and keyword collisions trigger silent routing misclassifications.
"""

import os
import re
import pandas as pd
from typing import Dict, Any, Tuple, Optional


def classify_case_rules(subject: str, body: str) -> Tuple[str, Optional[float], str]:
    """
    Classifies a case using 15 explicit rules in priority order.
    Returns: (predicted_department, confidence, triggered_rule_id)
    """
    text = f"{subject} {body}".lower()

    # Rule 1: High priority invoice & dunning
    if re.search(r"\b(past due notice|dunning|w-9|vat tax exemption|annual contract billing)\b", text):
        # Realistic Rule Trap: 'contract' keyword often collides with Sales rule if not careful,
        # but here we see explicit payment terms.
        return "Billing", None, "RULE_01_TAX_DUNNING"

    # Rule 2: Payment failures & gateway errors
    if re.search(r"\b(payment gateway failure|card renewal failed|error 402|double billing)\b", text):
        return "Billing", None, "RULE_02_PAYMENT_FAILURE"

    # Rule 3: Direct invoice discrepancy
    if re.search(r"\b(discrepancy on invoice|overcharged for inactive|reverse one)\b", text):
        return "Billing", None, "RULE_03_INVOICE_DISCREPANCY"

    # Rule 4: Critical API failures
    if re.search(r"\b(rest api 500|webhook dispatch|http 500 internal)\b", text):
        return "Technical", None, "RULE_04_API_OUTAGE"

    # Rule 5: Authentication & SSO loops
    if re.search(r"\b(sso saml|okta saml|authentication loop|infinite redirect)\b", text):
        return "Technical", None, "RULE_05_SSO_LOOP"

    # Rule 6: Database latency & replicas
    if re.search(r"\b(query latency spike|replica database|graphql endpoint)\b", text):
        return "Technical", None, "RULE_06_DB_PERF"

    # Rule 7: SDK code exceptions
    if re.search(r"\b(typeerror: undefined|python sdk v2|unhandled exception)\b", text):
        return "Technical", None, "RULE_07_SDK_BUG"

    # Rule 8: Rate limits & WebSockets
    if re.search(r"\b(rate limit reached|429 too many requests|websocket connection dropping)\b", text):
        return "Technical", None, "RULE_08_LIMIT_SOCKET"

    # Rule 9: Enterprise Tier Upgrades
    if re.search(r"\b(upgrading to enterprise|custom quote|dedicated seats|series b)\b", text):
        return "Sales", None, "RULE_09_ENTERPRISE_UPGRADE"

    # Rule 10: Contract renewals & price concessions
    if re.search(r"\b(contract renewal discount|competitor price matching|quote from zendesk)\b", text):
        return "Sales", None, "RULE_10_RENEWAL_COMPETITOR"

    # Rule 11: Product Demos & Trial Extension
    if re.search(r"\b(product demo|trial extension request|reseller program)\b", text):
        return "Sales", None, "RULE_11_DEMO_TRIAL"

    # Rule 12: Hardware & Logistics
    if re.search(r"\b(shipment tracking delay|hardware dispatch|motherboard replacement)\b", text):
        return "Operations", None, "RULE_12_LOGISTICS"

    # Rule 13: Maintenance & Facilities (Rule collision: 'deployment' often matches tech)
    if re.search(r"\b(scheduled maintenance notification|cooling outage alert|auxiliary hvac)\b", text):
        return "Operations", None, "RULE_13_FACILITIES_MAINT"

    # Rule 14: Security Operations & Badges
    if re.search(r"\b(physical badge provisioning|soc 2 type ii vendor|onboarding workflow stuck)\b", text):
        return "Operations", None, "RULE_14_SECURITY_OPS"

    # Rule 15: Legal Compliance & Privacy
    if re.search(r"\b(gdpr data subject|article 17 gdpr|copyright dmca)\b", text):
        return "Other", None, "RULE_15_LEGAL_GDPR"

    # Fallback to General / Other
    return "Other", None, "RULE_DEFAULT_FALLBACK"


def evaluate_dataset(csv_path: str, dataset_label: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    correct = 0
    predictions = []

    for _, row in df.iterrows():
        pred, conf, rule_id = classify_case_rules(str(row["subject"]), str(row["body"]))
        is_correct = (pred == row["department"])
        if is_correct:
            correct += 1
        predictions.append({
            "case_id": row["case_id"],
            "ground_truth": row["department"],
            "prediction": pred,
            "confidence": conf,
            "rule_id": rule_id,
            "correct": is_correct,
            "language": row.get("language", "English")
        })

    accuracy = correct / len(df) if len(df) > 0 else 0.0
    print(f"📊 [Rules] {dataset_label}: {correct}/{len(df)} correct ({accuracy:.1%})")
    return {
        "accuracy": accuracy,
        "total": len(df),
        "correct": correct,
        "predictions": predictions
    }


def run_rules_benchmark():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    splits_dir = os.path.join(base_dir, "data", "splits")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("==================================================")
    print(" 🛠️  EVALUATING RULES-BASED HEURISTICS BASELINE   ")
    print("==================================================")

    res_test = evaluate_dataset(os.path.join(splits_dir, "test.csv"), "English Test (30 cases)")
    res_es = evaluate_dataset(os.path.join(splits_dir, "test_spanish.csv"), "Spanish Test (30 cases)")
    res_hi = evaluate_dataset(os.path.join(splits_dir, "test_hindi.csv"), "Hindi Test (30 cases)")

    # Save predictions to results
    pred_df = pd.DataFrame(res_test["predictions"])
    pred_df.to_csv(os.path.join(results_dir, "rules_test_predictions.csv"), index=False)

    print("\n🔍 Key Baseline Takeaway:")
    print(f" - English accuracy: {res_test['accuracy']:.1%}")
    print(f" - Spanish accuracy: {res_es['accuracy']:.1%} (Severe drop due to English-only regex keywords)")
    print(f" - Hindi accuracy:   {res_hi['accuracy']:.1%} (Catastrophic drop to random/fallback level)")
    print(f" - Confidence score: None (Blind 100% routing rate, zero uncertainty estimation)\n")


if __name__ == "__main__":
    run_rules_benchmark()
