"""
Benchmark Visualizer & Plot Generator
-------------------------------------
Generates charts from final measured 200-case hard CRM benchmark artifacts:
1. confidence_distribution.png: Measured confidence distribution (Zero-Shot vs Fine-Tuned Laya).
2. routing_simulation.png: Automation coverage vs routing accuracy trade-off curves.
3. difficulty_breakdown.png: Measured accuracy across difficulty tiers (easy, paraphrase, multi-topic, ambiguous).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DATA_DIR = os.path.join(REPO_ROOT, "data", "einstein_hard_benchmark")

# Clean styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_confidence_distribution():
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    zs_path = os.path.join(RESULTS_DIR, "predictions_zeroshot_hard_english_dict.csv")
    ft_path = os.path.join(RESULTS_DIR, "colab_finetune", "predictions_hard_seed_2026.csv")

    if not (os.path.exists(zs_path) and os.path.exists(ft_path)):
        print("Missing prediction CSVs for confidence distribution plot.")
        return

    zs_confs = pd.read_csv(zs_path)["confidence"]
    ft_confs = pd.read_csv(ft_path)["confidence"]

    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(zs_confs, bins=bins, alpha=0.55, color="#ef4444", label=f"Laya Zero-Shot (Mean Conf = {zs_confs.mean():.3f})", edgecolor="white")
    ax.hist(ft_confs, bins=bins, alpha=0.65, color="#10b981", label=f"Laya Fine-Tuned (Mean Conf = {ft_confs.mean():.3f})", edgecolor="white")

    ax.set_title("Laya Confidence Distribution on Hard Benchmark (Zero-Shot vs GPU Fine-Tuned)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Calibrated Posterior Confidence Score P(y|x)", fontsize=10)
    ax.set_ylabel("Number of Test Cases (out of 200)", fontsize=10)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "confidence_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Saved: {out_path}")


def plot_routing_simulation():
    truth_path = os.path.join(DATA_DIR, "test_truth.csv")
    if not os.path.exists(truth_path):
        return
    truth = pd.read_csv(truth_path)

    series_config = [
        ("Laya Fine-Tuned (GPU)", os.path.join(RESULTS_DIR, "colab_finetune", "predictions_hard_seed_2026.csv"), "confidence", "predicted_department", "#059669", "-", "s"),
        ("TF-IDF + MLP (Broader 800)", os.path.join(RESULTS_DIR, "predictions_tfidf_origtrain_hardtest_50epochs.csv"), "confidence", "predicted_department", "#2563eb", "-", "o"),
        ("Laya Zero-Shot (Dict)", os.path.join(RESULTS_DIR, "predictions_zeroshot_hard_english_dict.csv"), "confidence", "predicted_department", "#d97706", "--", "^"),
        ("TF-IDF + MLP (Narrow 800)", os.path.join(RESULTS_DIR, "predictions_tfidf_hard.csv"), "confidence", "predicted_department", "#dc2626", ":", "x"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

    for label, path, conf_col, pred_col, color, linestyle, marker in series_config:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "true_department" not in df.columns:
            df = df.merge(truth[["ExternalId__c", "true_department"]], on="ExternalId__c")
        df["correct"] = (df[pred_col] == df["true_department"]).astype(int)
        df = df.sort_values(conf_col, ascending=False).reset_index(drop=True)
        df["cum_correct"] = df["correct"].cumsum()
        df["cov"] = df.index + 1
        df["acc"] = df["cum_correct"] / df["cov"] * 100.0

        # Subsample points for cleaner plot
        pts = df[df["cov"] >= 50]
        step = max(1, len(pts) // 15)
        pts_sampled = pts.iloc[::step]
        if pts.iloc[-1]["cov"] not in pts_sampled["cov"].values:
            pts_sampled = pd.concat([pts_sampled, pts.iloc[[-1]]])

        ax.plot(pts_sampled["cov"], pts_sampled["acc"], label=label, color=color, linestyle=linestyle, marker=marker, linewidth=2, markersize=5)

    ax.set_title("Confidence-Gated Routing: Coverage vs. Accuracy Trade-Off (Hard 200 Test Set)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Cases Auto-Routed (out of 200 Blind Test Cases)", fontsize=10)
    ax.set_ylabel("Routing Accuracy Among Auto-Routed Cases (%)", fontsize=10)
    ax.set_ylim(25, 105)
    ax.set_xlim(45, 205)
    ax.legend(frameon=True, facecolor="white", loc="lower left", fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "routing_simulation.png")
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Saved: {out_path}")


def plot_difficulty_breakdown():
    truth_path = os.path.join(DATA_DIR, "test_truth.csv")
    if not os.path.exists(truth_path):
        return
    truth = pd.read_csv(truth_path)

    tiers = ["easy", "paraphrase", "multi_topic", "ambiguous"]
    tier_labels = ["Easy (80)", "Paraphrase (60)", "Multi-Topic (40)", "Ambiguous (20)"]

    models = [
        ("Laya Fine-Tuned", os.path.join(RESULTS_DIR, "colab_finetune", "predictions_hard_seed_2026.csv"), "predicted_department", "#10b981"),
        ("TF-IDF (Broader)", os.path.join(RESULTS_DIR, "predictions_tfidf_origtrain_hardtest_50epochs.csv"), "predicted_department", "#3b82f6"),
        ("Laya Zero-Shot", os.path.join(RESULTS_DIR, "predictions_zeroshot_hard_english_dict.csv"), "predicted_department", "#f59e0b"),
        ("Keyword Rules", os.path.join(RESULTS_DIR, "predictions_rules_trained_hard.csv"), "predicted_department", "#94a3b8"),
        ("TF-IDF (Narrow)", os.path.join(RESULTS_DIR, "predictions_tfidf_hard.csv"), "predicted_department", "#ef4444"),
    ]

    data_by_model = {}
    for name, path, pred_col, color in models:
        if os.path.exists(path):
            df = pd.read_csv(path)
            cols_to_add = [c for c in ["true_department", "difficulty"] if c not in df.columns]
            if cols_to_add:
                df = df.merge(truth[["ExternalId__c"] + cols_to_add], on="ExternalId__c")
            df["correct"] = (df[pred_col] == df["true_department"]).astype(int)
            accs = [df[df["difficulty"] == t]["correct"].mean() * 100.0 for t in tiers]
            data_by_model[name] = (accs, color)

    # Einstein from sandbox export
    einstein_path = os.path.join(DATA_DIR, "einstein_hard_raw_export.csv")
    if os.path.exists(einstein_path):
        ein_df = pd.read_csv(einstein_path).merge(truth[["ExternalId__c", "true_department", "difficulty"]], on="ExternalId__c")
        ein_df["correct"] = (ein_df["Department__c"] == ein_df["true_department"]).astype(int)
        ein_accs = [ein_df[ein_df["difficulty"] == t]["correct"].mean() * 100.0 for t in tiers]
        data_by_model["Salesforce Einstein"] = (ein_accs, "#8b5cf6")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(tiers))
    n_models = len(data_by_model)
    width = 0.8 / n_models

    for i, (name, (accs, color)) in enumerate(data_by_model.items()):
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=name, color=color, alpha=0.9, edgecolor="white", linewidth=0.5)

    ax.set_title("Hard Benchmark Accuracy by Difficulty Tier (200 Test Cases)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, facecolor="white", loc="upper right", fontsize=8.5, ncol=2)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "difficulty_breakdown.png")
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Saved: {out_path}")


def generate_all_plots():
    print("=" * 80)
    print(" 🎨 GENERATING MEASURED BENCHMARK VISUALIZATIONS (HARD BENCHMARK)")
    print("=" * 80)
    plot_confidence_distribution()
    plot_routing_simulation()
    plot_difficulty_breakdown()
    print("✅ Benchmark charts updated successfully!\n")


if __name__ == "__main__":
    generate_all_plots()
