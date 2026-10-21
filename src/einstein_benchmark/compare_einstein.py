"""
Side-by-Side Comparison: Laya vs. Heuristic Rules vs. Salesforce Einstein
--------------------------------------------------------------------------
Phase 2 - Step 5:
Ingests einstein_predictions.csv (ExternalId__c, Einstein_Department, Einstein_Confidence).
Compares predictions against Laya (laya_predictions.csv) and Rules on test_truth.csv.

CRITICAL BENCHMARK INTEGRITY RULE:
If einstein_predictions.csv is missing, exits cleanly with a waiting message.
NEVER invents, estimates, or mocks Einstein metrics.
"""

import os
import sys
import pandas as pd
import numpy as np

# Import Rules baseline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.rules import classify_case_rules

DEPARTMENTS = ["Billing", "Technical", "Sales", "Operations", "Other"]
DIFFICULTIES = ["easy", "paraphrase", "multi_topic", "ambiguous"]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def compare_models():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(base_dir, "data", "einstein_benchmark")

    einstein_path = os.path.join(data_dir, "einstein_predictions.csv")
    laya_path = os.path.join(data_dir, "laya_predictions.csv")
    truth_path = os.path.join(data_dir, "test_truth.csv")
    blind_path = os.path.join(data_dir, "test_blind.csv")

    # Step 5 Rule: Check for einstein_predictions.csv
    if not os.path.exists(einstein_path):
        print("=" * 70)
        print(" ⏳ WAITING FOR EINSTEIN PREDICTIONS")
        print("=" * 70)
        print("Waiting for einstein_predictions.csv. Once Einstein finishes training")
        print("in your org, export predictions on test cases and save them here:")
        print(f" -> {einstein_path}")
        print("Required CSV columns: ExternalId__c, Einstein_Department, Einstein_Confidence")
        print("=" * 70)
        sys.exit(0)

    print("=" * 80)
    print(" 🏆 HEAD-TO-HEAD BENCHMARK: LAYA vs. RULES vs. SALESFORCE EINSTEIN")
    print("=" * 80)

    # Load truth
    truth_df = pd.read_csv(truth_path)
    blind_df = pd.read_csv(blind_path)
    test_df = pd.merge(blind_df, truth_df, on="ExternalId__c")

    # Load Laya
    if not os.path.exists(laya_path):
        print(f"❌ Error: Laya predictions missing at {laya_path}. Run train_and_evaluate_laya.py first.")
        sys.exit(1)
    laya_df = pd.read_csv(laya_path)

    # Load Einstein
    einstein_df = pd.read_csv(einstein_path)
    required_cols = {"ExternalId__c", "Einstein_Department", "Einstein_Confidence"}
    if not required_cols.issubset(set(einstein_df.columns)):
        print(f"❌ Error: einstein_predictions.csv must contain columns: {required_cols}")
        sys.exit(1)

    # Compute Rules predictions
    rules_preds = []
    for _, row in test_df.iterrows():
        r_pred, _, _ = classify_case_rules(str(row["Subject"]), str(row["Description"]))
        rules_preds.append(r_pred)

    merged = test_df.merge(laya_df, on="ExternalId__c").merge(einstein_df, on="ExternalId__c")
    total_cases = len(merged)

    # Correctness flags
    merged["rules_correct"] = (np.array(rules_preds) == merged["true_department"])
    merged["laya_correct"] = (merged["predicted_department"] == merged["true_department"])
    merged["einstein_correct"] = (merged["Einstein_Department"] == merged["true_department"])

    # 1. Overall & Difficulty Breakdown
    print(f"\n1. ACCURACY BY DIFFICULTY TIER ({total_cases} Unseen Cases)")
    print("-" * 80)
    print(f"{'Tier':<14} | {'Count':<6} | {'Rules':<16} | {'Einstein':<16} | {'Laya':<16}")
    print("-" * 80)

    r_all = merged['rules_correct'].mean()
    e_all = merged['einstein_correct'].mean()
    l_all = merged['laya_correct'].mean()
    print(f"{'OVERALL':<14} | {total_cases:<6} | {merged['rules_correct'].sum()}/{total_cases} ({r_all:.1%})  | {merged['einstein_correct'].sum()}/{total_cases} ({e_all:.1%})  | {merged['laya_correct'].sum()}/{total_cases} ({l_all:.1%})")
    print("-" * 80)

    for diff in DIFFICULTIES:
        sub = merged[merged["difficulty"] == diff]
        n = len(sub)
        r_c = sub["rules_correct"].sum()
        e_c = sub["einstein_correct"].sum()
        l_c = sub["laya_correct"].sum()
        r_p = r_c / n if n > 0 else 0
        e_p = e_c / n if n > 0 else 0
        l_p = l_c / n if n > 0 else 0
        print(f"{diff:<14} | {n:<6} | {r_c}/{n} ({r_p:.1%}){' ':<5} | {e_c}/{n} ({e_p:.1%}){' ':<5} | {l_c}/{n} ({l_p:.1%})")
    print("-" * 80)

    # 2. Confidence Gating Comparison (Laya vs. Einstein)
    print("\n2. CONFIDENCE GATING THRESHOLD SWEEP (tau in [0.50, 0.95])")
    print("-" * 80)
    print(f"{'Threshold (tau)':<16} | {'Laya Auto % (Acc)':<24} | {'Einstein Auto % (Acc)':<24}")
    print("-" * 80)

    for tau in THRESHOLDS:
        l_sub = merged[merged["confidence"] >= tau]
        e_sub = merged[merged["Einstein_Confidence"] >= tau]

        l_pct = (len(l_sub) / total_cases) * 100.0
        e_pct = (len(e_sub) / total_cases) * 100.0

        l_acc_str = f"{l_sub['laya_correct'].mean() * 100.0:.1f}%" if len(l_sub) > 0 else "N/A"
        e_acc_str = f"{e_sub['einstein_correct'].mean() * 100.0:.1f}%" if len(e_sub) > 0 else "N/A"

        l_cell = f"{l_pct:5.1f}% ({l_acc_str})"
        e_cell = f"{e_pct:5.1f}% ({e_acc_str})"

        print(f"{tau:<16.2f} | {l_cell:<24} | {e_cell:<24}")
    print("=" * 80)


if __name__ == "__main__":
    compare_models()
