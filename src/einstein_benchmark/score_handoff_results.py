"""
Score Handoff Results (Colab Fine-Tuning & Sandbox Einstein Hard Run)
=====================================================================
Scores all empirical results directly from committed prediction files and ground truth.
No hardcoded or typed-in numbers.

Evaluates:
1. Laya GPU Fine-Tuned (ModernBERT-large, seeds 2026, 2027, 2028):
   - Overall Top-1 accuracy + Wilson 95% CI
   - Expected Calibration Error (ECE)
   - Confidence-threshold gating (thresholds >= 0.99 and >= 0.90) with routed case counts, accuracy, and CI
2. Salesforce Einstein Case Classification (Sandbox Hard Run):
   - Scored from data/einstein_hard_benchmark/einstein_hard_raw_export.csv
   - Automated routing (confidence threshold >= 90 configured in org)
   - Overall Top-1 accuracy + Wilson 95% CI
   - Routed accuracy + Wilson 95% CI
3. Training set lexical diversity audit:
   - Evaluates unique raw descriptions and unique descriptions after masking digits.
"""

import os
import sys
import json
import math
import re
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)


def wilson_interval(successes: int, total: int):
    if total == 0:
        return 0.0, 0.0, 0.0
    z = 1.959963984540054  # 95% confidence
    p_hat = successes / total
    denom = 1 + (z**2) / total
    center = (p_hat + (z**2) / (2 * total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * total)) / total) / denom
    return p_hat, max(0.0, center - spread), min(1.0, center + spread)


def compute_ece(confs, accs, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confs)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (confs > bin_lower) & (confs <= bin_upper) if i > 0 else (confs >= bin_lower) & (confs <= bin_upper)
        if np.sum(mask) > 0:
            bin_acc = np.mean(accs[mask])
            bin_conf = np.mean(confs[mask])
            ece += (np.sum(mask) / total) * np.abs(bin_acc - bin_conf)
    return float(ece)


def score_laya_predictions(pred_path: str):
    df = pd.read_csv(pred_path)
    total = len(df)
    corr = int((df["predicted_department"] == df["true_department"]).sum())
    p_ovr, l_ovr, u_ovr = wilson_interval(corr, total)
    ece = compute_ece(df["confidence"].values, (df["predicted_department"] == df["true_department"]).values.astype(float))

    threshold_results = {}
    for tau in [0.99, 0.90]:
        gated = df[df["confidence"] >= tau]
        g_count = len(gated)
        g_corr = int((gated["predicted_department"] == gated["true_department"]).sum()) if g_count > 0 else 0
        p_g, l_g, u_g = wilson_interval(g_corr, g_count) if g_count > 0 else (0.0, 0.0, 0.0)
        threshold_results[str(tau)] = {
            "threshold": tau,
            "cases_routed": g_count,
            "coverage_pct": g_count / total,
            "routed_correct": g_corr,
            "routed_accuracy": p_g,
            "routed_ci": [l_g, u_g]
        }

    return {
        "total_cases": total,
        "overall_correct": corr,
        "overall_accuracy": p_ovr,
        "overall_ci": [l_ovr, u_ovr],
        "ece": round(ece, 4),
        "confidence_gating": threshold_results
    }


def score_einstein_hard(raw_export_path: str, truth_path: str):
    raw = pd.read_csv(raw_export_path)
    truth = pd.read_csv(truth_path)
    m = pd.merge(raw, truth, on="ExternalId__c")
    total = len(m)

    m["pred_dept"] = m["Department__c"].fillna("")
    routed = m[m["pred_dept"] != ""]
    unrouted_count = int((m["pred_dept"] == "").sum())

    routed_count = len(routed)
    routed_corr = int((routed["pred_dept"] == routed["true_department"]).sum())
    p_rout, l_rout, u_rout = wilson_interval(routed_corr, routed_count) if routed_count > 0 else (0.0, 0.0, 0.0)

    overall_corr = int((m["pred_dept"] == m["true_department"]).sum())
    p_ovr, l_ovr, u_ovr = wilson_interval(overall_corr, total)

    return {
        "model": "Salesforce Einstein Case Classification (Sandbox)",
        "training_set": "Hard 800 (train_salesforce_import.csv)",
        "test_set": "Hard 200 (test_salesforce_import.csv)",
        "total_cases": total,
        "overall_correct": overall_corr,
        "overall_accuracy": p_ovr,
        "overall_ci": [l_ovr, u_ovr],
        "cases_routed": routed_count,
        "cases_unrouted": unrouted_count,
        "coverage_pct": routed_count / total,
        "routed_correct": routed_corr,
        "routed_accuracy": p_rout,
        "routed_ci": [l_rout, u_rout],
        "configuration": {
            "automate_value": "On",
            "confidence_threshold": 90,
            "select_value": "Select Best (threshold 0)"
        }
    }


def check_training_descriptions(train_path: str):
    df = pd.read_csv(train_path)
    raw_desc_unique = df["Description"].nunique()
    masked = df["Description"].apply(lambda s: re.sub(r"\d+", "<NUM>", str(s).strip().lower()))
    masked_desc_unique = masked.nunique()
    return {
        "total_cases": len(df),
        "raw_unique_descriptions": raw_desc_unique,
        "masked_unique_descriptions": masked_desc_unique
    }


def main():
    colab_dir = os.path.join(REPO_ROOT, "results", "colab_finetune")
    hard_dir = os.path.join(REPO_ROOT, "data", "einstein_hard_benchmark")

    truth_hard_path = os.path.join(hard_dir, "test_truth.csv")
    einstein_raw_path = os.path.join(hard_dir, "einstein_hard_raw_export.csv")
    train_hard_path = os.path.join(hard_dir, "train_closed.csv")

    # 1. Training set description audit
    desc_audit = check_training_descriptions(train_hard_path)
    print("=" * 80)
    print(f"TRAINING SET LEXICAL DIVERSITY AUDIT ({train_hard_path}):")
    print(f"  Total cases: {desc_audit['total_cases']}")
    print(f"  Raw unique descriptions: {desc_audit['raw_unique_descriptions']}")
    print(f"  Masked unique descriptions (digits -> <NUM>): {desc_audit['masked_unique_descriptions']}")
    print("=" * 80)

    # 2. Score Einstein on Hard Benchmark
    einstein_res = score_einstein_hard(einstein_raw_path, truth_hard_path)
    print("\n" + "=" * 80)
    print("SALESFORCE EINSTEIN (HARD 800 TRAINING -> HARD 200 TEST):")
    print(f"  Total Cases: {einstein_res['total_cases']}")
    print(f"  Overall Top-1 Accuracy: {einstein_res['overall_correct']}/{einstein_res['total_cases']} = {einstein_res['overall_accuracy']*100:.2f}% [{einstein_res['overall_ci'][0]*100:.1f}%, {einstein_res['overall_ci'][1]*100:.1f}%]")
    print(f"  Cases Routed (conf >= 90): {einstein_res['cases_routed']}/{einstein_res['total_cases']} ({einstein_res['coverage_pct']*100:.1f}% coverage, {einstein_res['cases_unrouted']} unrouted)")
    print(f"  Routed Accuracy: {einstein_res['routed_correct']}/{einstein_res['cases_routed']} = {einstein_res['routed_accuracy']*100:.2f}% [{einstein_res['routed_ci'][0]*100:.1f}%, {einstein_res['routed_ci'][1]*100:.1f}%]")
    print("=" * 80)

    # 3. Score Laya Fine-Tuned Seeds
    seeds = [2026, 2027, 2028]
    laya_results = {"hard": {}, "orig": {}}

    for test_key in ["hard", "orig"]:
        print("\n" + "=" * 80)
        print(f"LAYA GPU FINE-TUNED RUNS (TEST SET: {test_key.upper()} 200 CASES):")
        print("=" * 80)
        seed_accs = []
        seed_r99_accs = []
        seed_r90_accs = []
        seed_r99_covs = []
        seed_r90_covs = []
        seed_eces = []

        for s in seeds:
            csv_path = os.path.join(colab_dir, f"predictions_{test_key}_seed_{s}.csv")
            s_res = score_laya_predictions(csv_path)
            laya_results[test_key][s] = s_res
            seed_accs.append(s_res["overall_accuracy"])
            seed_eces.append(s_res["ece"])

            r99 = s_res["confidence_gating"]["0.99"]
            seed_r99_accs.append(r99["routed_accuracy"])
            seed_r99_covs.append(r99["coverage_pct"])

            r90 = s_res["confidence_gating"]["0.9"]
            seed_r90_accs.append(r90["routed_accuracy"])
            seed_r90_covs.append(r90["coverage_pct"])

            print(f"Seed {s}:")
            print(f"  Overall: {s_res['overall_correct']}/200 = {s_res['overall_accuracy']*100:.2f}% [{s_res['overall_ci'][0]*100:.1f}%, {s_res['overall_ci'][1]*100:.1f}%], ECE={s_res['ece']:.4f}")
            print(f"  Routed @ conf >= 0.99: {r99['routed_correct']}/{r99['cases_routed']} = {r99['routed_accuracy']*100:.2f}% [{r99['routed_ci'][0]*100:.1f}%, {r99['routed_ci'][1]*100:.1f}%] (cov {r99['coverage_pct']*100:.1f}%)")
            print(f"  Routed @ conf >= 0.90: {r90['routed_correct']}/{r90['cases_routed']} = {r90['routed_accuracy']*100:.2f}% [{r90['routed_ci'][0]*100:.1f}%, {r90['routed_ci'][1]*100:.1f}%] (cov {r90['coverage_pct']*100:.1f}%)")

        laya_results[test_key]["summary"] = {
            "mean_overall_accuracy": float(np.mean(seed_accs)),
            "std_overall_accuracy": float(np.std(seed_accs)),
            "mean_ece": float(np.mean(seed_eces)),
            "mean_routed_acc_0_99": float(np.mean(seed_r99_accs)),
            "mean_coverage_0_99": float(np.mean(seed_r99_covs)),
            "mean_routed_acc_0_90": float(np.mean(seed_r90_accs)),
            "mean_coverage_0_90": float(np.mean(seed_r90_covs)),
        }
        summ = laya_results[test_key]["summary"]
        print(f"\nMean Overall Top-1 ({test_key.upper()}): {summ['mean_overall_accuracy']*100:.2f}% ± {summ['std_overall_accuracy']*100:.2f}%")
        print(f"Mean Routed @ conf >= 0.99: {summ['mean_routed_acc_0_99']*100:.2f}% (mean coverage {summ['mean_coverage_0_99']*100:.1f}%)")
        print(f"Mean Routed @ conf >= 0.90: {summ['mean_routed_acc_0_90']*100:.2f}% (mean coverage {summ['mean_coverage_0_90']*100:.1f}%)")

    # 4. Save JSON summary
    summary_path = os.path.join(REPO_ROOT, "results", "handoff_scoring_summary.json")
    full_output = {
        "training_description_caveat": desc_audit,
        "salesforce_einstein_hard": einstein_res,
        "laya_fine_tuned": laya_results
    }
    with open(summary_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nSaved full scoring summary to {summary_path}")


if __name__ == "__main__":
    main()
