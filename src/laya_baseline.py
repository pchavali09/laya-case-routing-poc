"""
Laya Baseline (Zero-Shot) Classifier
-------------------------------------
Evaluates the uncalibrated zero-shot performance of real Laya (convaiinnovations/laya)
on the 200 blind test cases from data/einstein_benchmark/test_blind.csv.

Uses Laya's documented API: Router().predict(state, questions) with choice questions.
No hardcoded indices, no synthetic matrix, and no silent fallback.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from laya import Router

DEPARTMENTS = ["Billing", "Technical", "Sales", "Operations", "Other"]


def run_zero_shot_inference(
    test_blind_path: str,
    test_truth_path: str,
    criteria_mode: str = "list"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes real Laya zero-shot routing across the 200 benchmark test cases.
    Raises RuntimeError immediately if Laya fails to load or predict (no fallback).
    """
    print(f"Loading real Laya model (convaiinnovations/laya) with default='english'...")
    try:
        router = Router(default="english")
    except Exception as e:
        raise RuntimeError(f"Failed to load real Laya model: {e}") from e

    test_blind = pd.read_csv(test_blind_path)
    test_truth = pd.read_csv(test_truth_path)
    merged = pd.merge(test_blind, test_truth, on="ExternalId__c")

    if criteria_mode == "dict":
        criteria = {
            "Billing": "invoices, payments, refunds, charges, subscriptions, pricing",
            "Technical": "system errors, software bugs, outages, API integration, technical issues",
            "Sales": "quotes, enterprise licensing, contract renewals, upgrades, sales demos",
            "Operations": "shipping, logistics, warehouse, facilities, hardware procurement",
            "Other": "general inquiries, uncategorized corporate communications, miscellaneous feedback"
        }
    else:
        criteria = DEPARTMENTS

    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this customer support case?",
            "criteria": criteria
        }
    }

    # Warmup call
    router.predict("Warmup system check", questions)

    predictions = []
    latencies = []
    t_start = time.time()

    for _, row in merged.iterrows():
        state = f"Subject: {row['Subject']}\nDescription: {row['Description']}"
        t0 = time.time()
        try:
            res = router.predict(state, questions)
        except Exception as e:
            raise RuntimeError(f"Laya predict failed on case {row['ExternalId__c']}: {e}") from e
        lat_ms = (time.time() - t0) * 1000.0
        latencies.append(lat_ms)

        ans = res["answers"]["department"]
        pred_choice = ans["choice"]
        conf = float(ans.get("confidence", ans.get("answer_confidence", 0.0)))

        predictions.append({
            "ExternalId__c": row["ExternalId__c"],
            "true_department": row["true_department"],
            "predicted_department": pred_choice,
            "confidence": round(conf, 4),
            "correct": (pred_choice == row["true_department"]),
            "latency_ms": round(lat_ms, 2),
            "difficulty": row["difficulty"],
            "model_artifact": "convaiinnovations/laya"
        })

    total_time = time.time() - t_start
    pred_df = pd.DataFrame(predictions)
    acc = pred_df["correct"].mean()

    metrics = {
        "total_cases": len(pred_df),
        "correct": int(pred_df["correct"].sum()),
        "accuracy": round(float(acc) * 100, 2),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "median_latency_ms": round(float(np.median(latencies)), 2),
        "total_time_s": round(total_time, 2),
        "criteria_mode": criteria_mode,
        "model_artifact": "convaiinnovations/laya"
    }

    return pred_df, metrics


def main():
    test_blind_path = os.path.join(BASE_DIR, "data", "einstein_benchmark", "test_blind.csv")
    test_truth_path = os.path.join(BASE_DIR, "data", "einstein_benchmark", "test_truth.csv")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("==================================================")
    print(" 🤖 EVALUATING REAL LAYA BASELINE (ZERO-SHOT)     ")
    print("==================================================")

    # Run list criteria
    pred_df_list, metrics_list = run_zero_shot_inference(test_blind_path, test_truth_path, "list")
    out_csv = os.path.join(results_dir, "laya_zeroshot_predictions.csv")
    pred_df_list.to_csv(out_csv, index=False)

    print(f"\n📊 [Real Laya Zero-Shot (List Criteria)] 200 Test Cases:")
    print(f" - Top-1 Accuracy: {metrics_list['correct']}/{metrics_list['total_cases']} ({metrics_list['accuracy']}%)")
    print(f" - Mean Latency: {metrics_list['mean_latency_ms']} ms")
    print(f" - Saved predictions to {out_csv}\n")


if __name__ == "__main__":
    from typing import Tuple
    main()
