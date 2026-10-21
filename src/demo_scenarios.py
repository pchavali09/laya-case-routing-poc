"""
Interactive Scenario Runner: Rules vs. Laya Side-by-Side
--------------------------------------------------------
Executes 5 real-world scenarios comparing the rule-based heuristics against
Laya's calibrated local decision engine.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rules import classify_case_rules
from src.laya_inference import LayaInferenceEngine

def run_scenarios():
    engine = LayaInferenceEngine()

    test_scenarios = [
        {
            "title": "Scenario 1: Clear Billing Issue (English)",
            "subject": "Double billing detected on corporate card",
            "body": "Our finance department flagged two identical charges of $1250 posted on the same day. Please reverse one.",
            "ground_truth": "Billing"
        },
        {
            "title": "Scenario 2: Critical Technical Outage (English)",
            "subject": "REST API 500 error on webhook dispatch",
            "body": "Since 09:00 UTC, all outbound webhook events to our ingestion endpoint fail with HTTP 500 Internal Server Error. Logs attached.",
            "ground_truth": "Technical"
        },
        {
            "title": "Scenario 3: Spanish Language Billing (Multilingual Test)",
            "subject": "Discrepancia en la factura #48201",
            "body": "Notamos un cargo inesperado de $499 en la factura de nuestra suscripción empresarial. Por favor aclarar antes del pago.",
            "ground_truth": "Billing"
        },
        {
            "title": "Scenario 4: Hindi Language Technical (Multilingual Test)",
            "subject": "वेबहूक प्रेषण पर REST API 500 त्रुटि",
            "body": "सुबह 09:00 UTC से, हमारे समापन बिंदु पर सभी आउटबाउंड वेबहूक HTTP 500 आंतरिक त्रुटि के साथ विफल हो रहे हैं।",
            "ground_truth": "Technical"
        },
        {
            "title": "Scenario 5: Ambiguous / Borderline SLA Case (Edge Case)",
            "subject": "Custom SLA terms for dedicated hosting",
            "body": "Our compliance team requires 99.99% uptime with financial penalty clauses. Can we negotiate custom SLA terms under our contract renewal?",
            "ground_truth": "Sales"
        }
    ]

    print("====================================================================================================")
    print("  SIDE-BY-SIDE EVALUATION: RULES-BASED HEURISTICS vs. LAYA CALIBRATED ENGINE")
    print("====================================================================================================\n")

    for sc in test_scenarios:
        print(f"📌 {sc['title']}")
        print(f"   Subject: \"{sc['subject']}\"")
        print(f"   Body:    \"{sc['body']}\"")
        print(f"   Ground Truth: {sc['ground_truth']}")
        print("   " + "-"*96)
        
        # 1. Rules Baseline
        t0 = time.perf_counter()
        rule_pred, rule_conf, rule_id = classify_case_rules(sc["subject"], sc["body"])
        rule_time = (time.perf_counter() - t0) * 1000.0
        rule_status = "✅ CORRECT" if rule_pred == sc["ground_truth"] else "❌ MISROUTED"

        print(f"   [RULES BASELINE]")
        print(f"     Prediction:  {rule_pred} ({rule_status})")
        print(f"     Confidence:  {rule_conf} (None - blind routing)")
        print(f"     Action:      AUTO_ROUTE (Blind dispatch, zero uncertainty estimation)")
        print(f"     Trigger:     {rule_id}")
        print(f"     Latency:     {rule_time:.3f} ms")
        
        # 2. Laya Local Engine
        t0 = time.perf_counter()
        laya_res = engine.classify_case(sc["subject"], sc["body"], confidence_threshold=0.85)
        laya_time = (time.perf_counter() - t0) * 1000.0
        laya_status = "✅ CORRECT" if laya_res["predicted_department"] == sc["ground_truth"] else "❌ MISROUTED"

        print(f"   [LAYA LOCAL-FIRST ENGINE]")
        print(f"     Prediction:  {laya_res['predicted_department']} ({laya_status})")
        print(f"     Confidence:  {laya_res['confidence']:.2%} (Calibrated posterior P(y|x))")
        print(f"     Decision:    {laya_res['decision']}")
        print(f"     Reason:      {laya_res['routing_reason']}")
        print(f"     Language:    {laya_res['language']}")
        print(f"     Top Probs:   {dict(list(laya_res['probabilities'].items())[:3])}")
        print(f"     Latency:     {laya_time:.3f} ms")
        print("="*100 + "\n")

if __name__ == "__main__":
    run_scenarios()
