"""
System Comparison: Rules vs. Salesforce Einstein vs. Laya Foundation Models
---------------------------------------------------------------------------
Empirical comparison on the 200-case hard CRM benchmark (800 narrow train, 200 hard test).
Dynamically reads all values from results/handoff_scoring_summary.json and per-case CSVs.
"""

import os
import sys
import json
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DATA_DIR = os.path.join(REPO_ROOT, "data", "einstein_hard_benchmark")


def load_file_or_die(path: str, desc: str):
    if not os.path.exists(path):
        print(f"Error: Required {desc} file missing at '{path}'", file=sys.stderr)
        sys.exit(1)


def score_csv(path: str, pred_col: str = "predicted_department", true_col: str = "true_department") -> float:
    load_file_or_die(path, "prediction CSV")
    df = pd.read_csv(path)
    if pred_col not in df.columns or true_col not in df.columns:
        print(f"Error: Missing expected columns '{pred_col}' or '{true_col}' in '{path}'", file=sys.stderr)
        sys.exit(1)
    correct = (df[pred_col] == df[true_col]).sum()
    return correct / len(df) * 100.0


def run_comprehensive_comparison():
    summary_path = os.path.join(RESULTS_DIR, "handoff_scoring_summary.json")
    load_file_or_die(summary_path, "handoff scoring summary")

    with open(summary_path) as f:
        handoff = json.load(f)

    # 1. Laya fine-tuned
    try:
        hard_summary = handoff["laya_fine_tuned"]["hard"]["summary"]
        hard_2026 = handoff["laya_fine_tuned"]["hard"]["2026"]["overall_accuracy"] * 100.0
        hard_2027 = handoff["laya_fine_tuned"]["hard"]["2027"]["overall_accuracy"] * 100.0
        hard_2028 = handoff["laya_fine_tuned"]["hard"]["2028"]["overall_accuracy"] * 100.0
        laya_ft_acc = f"{hard_summary['mean_overall_accuracy'] * 100.0:.1f}%"
        laya_ft_notes = f"Average of 3 seeds ({hard_2026:.1f}% / {hard_2027:.1f}% / {hard_2028:.1f}%)"
    except KeyError as e:
        print(f"Error: Missing expected Laya fine-tuned key {e} in '{summary_path}'", file=sys.stderr)
        sys.exit(1)

    # 2. TF-IDF + MLP (Broader 800)
    tfidf_broader_path = os.path.join(RESULTS_DIR, "predictions_tfidf_origtrain_hardtest_50epochs.csv")
    tfidf_broader_acc = f"{score_csv(tfidf_broader_path):.1f}%"

    # 3. Laya zero-shot
    hard_summary_path = os.path.join(RESULTS_DIR, "measured_benchmark_hard_summary.json")
    load_file_or_die(hard_summary_path, "hard benchmark summary")
    with open(hard_summary_path) as f:
        hard_bench_rows = json.load(f)

    zs_accs = {}
    for r in hard_bench_rows:
        model = r.get("model", "")
        if "Laya English Zero-Shot (Dict" in model:
            zs_accs["en_dict"] = r["overall_accuracy"] * 100.0
        elif "Laya English Zero-Shot (List" in model:
            zs_accs["en_list"] = r["overall_accuracy"] * 100.0
        elif "Laya Multilingual Zero-Shot (Dict" in model:
            zs_accs["multi_dict"] = r["overall_accuracy"] * 100.0
        elif "Laya Multilingual Zero-Shot (List" in model:
            zs_accs["multi_list"] = r["overall_accuracy"] * 100.0

    if len(zs_accs) < 4:
        print(f"Error: Could not extract all zero-shot variations from '{hard_summary_path}'", file=sys.stderr)
        sys.exit(1)

    zs_min = min(zs_accs.values())
    zs_max = max(zs_accs.values())
    laya_zs_acc = f"{zs_min:.0f}% to {zs_max:.0f}%"
    laya_zs_notes = f"{zs_accs['en_dict']:.1f}%–{zs_accs['en_list']:.1f}% (English), {zs_accs['multi_list']:.1f}%–{zs_accs['multi_dict']:.1f}% (Multilingual)"

    # 4. Keyword rules (Trained)
    rules_path = os.path.join(RESULTS_DIR, "predictions_rules_trained_hard.csv")
    rules_acc = f"{score_csv(rules_path):.1f}%"

    # 5. TF-IDF + MLP (Narrow 800)
    tfidf_narrow_path = os.path.join(RESULTS_DIR, "predictions_tfidf_hard.csv")
    tfidf_narrow_acc = f"{score_csv(tfidf_narrow_path):.1f}%"

    # 6. Salesforce Einstein
    try:
        ein = handoff["salesforce_einstein_hard"]
        ein_acc = f"{ein['overall_accuracy'] * 100.0:.1f}%"
        ci_low = ein["overall_ci"][0] * 100.0
        ci_high = ein["overall_ci"][1] * 100.0
        ein_notes = (
            f"{ein['overall_correct']} of {ein['total_cases']} right; "
            f"{ein['cases_routed']} routed, all to Billing; "
            f"{ein['cases_unrouted']} left unrouted; "
            f"95% CI {ci_low:.1f}% to {ci_high:.1f}%"
        )
    except KeyError as e:
        print(f"Error: Missing expected Salesforce Einstein key {e} in '{summary_path}'", file=sys.stderr)
        sys.exit(1)

    # Verify Einstein export matches Billing collapse
    ein_raw_path = os.path.join(DATA_DIR, "einstein_hard_raw_export.csv")
    load_file_or_die(ein_raw_path, "Einstein hard raw export")
    ein_df = pd.read_csv(ein_raw_path)
    routed_depts = set(ein_df[ein_df["Department__c"].notna() & (ein_df["Department__c"] != "")]["Department__c"].unique())
    if routed_depts != {"Billing"}:
        print(f"Warning: Expected Einstein routed cases to be only Billing, got {routed_depts}", file=sys.stderr)

    rows = [
        {
            "Model / System": "Laya fine-tuned",
            "Training Data": "Narrow 800 cases (GPU fine-tune)",
            "Top-1 Accuracy (200-Case Test)": laya_ft_acc,
            "Notes": laya_ft_notes,
        },
        {
            "Model / System": "TF-IDF + MLP",
            "Training Data": "Broader set of 800 cases",
            "Top-1 Accuracy (200-Case Test)": tfidf_broader_acc,
            "Notes": "Scored on the same 200-case test; only training data changed",
        },
        {
            "Model / System": "Laya zero-shot",
            "Training Data": "None (no fine-tuning)",
            "Top-1 Accuracy (200-Case Test)": laya_zs_acc,
            "Notes": laya_zs_notes,
        },
        {
            "Model / System": "Keyword rules",
            "Training Data": "Built from the 800 training cases only",
            "Top-1 Accuracy (200-Case Test)": rules_acc,
            "Notes": "Frozen rules tuned strictly on training cases (src/rules_trained.py)",
        },
        {
            "Model / System": "TF-IDF + MLP",
            "Training Data": "Same narrow 800 cases",
            "Top-1 Accuracy (200-Case Test)": tfidf_narrow_acc,
            "Notes": "Fails when test themes are absent from training (61/200)",
        },
        {
            "Model / System": "Salesforce Einstein Case Classification",
            "Training Data": "Sandbox model trained on narrow 800",
            "Top-1 Accuracy (200-Case Test)": ein_acc,
            "Notes": ein_notes,
        },
    ]

    print("=" * 80)
    print(" ⚖️  EMPIRICAL BENCHMARK COMPARISON (200-CASE HARD CRM BENCHMARK)")
    print("=" * 80)
    print("Test Set: 200 hard cases with 10 held-out themes absent from training.\n")

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_comprehensive_comparison()
