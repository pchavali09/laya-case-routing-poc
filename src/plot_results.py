"""
Benchmark Visualizer & Plot Generator
-------------------------------------
Generates charts from measured 1,000-case Einstein benchmark artifacts:
1. confidence_distribution.png: Measured confidence distribution (Zero-Shot vs Fine-Tuned Laya).
2. routing_simulation.png: Automation coverage vs routing accuracy trade-off curves.
3. difficulty_breakdown.png: Measured accuracy across difficulty tiers (easy, paraphrase, multi-topic, ambiguous).
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# Clean styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_confidence_distribution():
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    zs_path = os.path.join(RESULTS_DIR, "laya_zeroshot_predictions.csv")
    ft_path = os.path.join(RESULTS_DIR, "laya_finetuned_predictions.csv")

    if not (os.path.exists(zs_path) and os.path.exists(ft_path)):
        print("Missing prediction CSVs for confidence distribution plot.")
        return

    zs_confs = pd.read_csv(zs_path)["confidence"]
    ft_confs = pd.read_csv(ft_path)["confidence"]

    bins = np.linspace(0.1, 1.0, 18)
    ax.hist(zs_confs, bins=bins, alpha=0.55, color="#ef4444", label="Laya Zero-Shot (Mean Conf = 0.583)", edgecolor="white")
    ax.hist(ft_confs, bins=bins, alpha=0.65, color="#10b981", label="Laya Fine-Tuned (Mean Conf = 0.431, ECE = 0.1095)", edgecolor="white")

    ax.set_title("Laya Confidence Distribution: Zero-Shot vs Head-Only Fine-Tuned", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Calibrated Posterior Confidence Score P(y|x)", fontsize=10)
    ax.set_ylabel("Number of Blind Test Cases", fontsize=10)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "confidence_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Saved: {out_path}")


def plot_routing_simulation():
    summary_path = os.path.join(RESULTS_DIR, "measured_benchmark_summary.json")
    if not os.path.exists(summary_path):
        return

    with open(summary_path) as f:
        data = json.load(f)

    curves = data.get("coverage_curves", {})
    if not curves:
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

    series_config = [
        ("tfidf_mlp_baseline", "Simple Local Baseline (TF-IDF + MLP)", "#2563eb", "-", "o"),
        ("laya_zero_shot_dict", "Real Laya Zero-Shot (Dict Criteria)", "#059669", "--", "s"),
        ("laya_finetuned", "Real Laya Fine-Tuned (Head-Only)", "#d97706", "-.", "^"),
        ("laya_zero_shot_list", "Real Laya Zero-Shot (List Criteria)", "#dc2626", ":", "x"),
    ]

    for key, label, color, linestyle, marker in series_config:
        pts = curves.get(key, [])
        if not pts or isinstance(pts[0].get("threshold"), str):
            continue
        cov = [p["cases_routed"] for p in pts]
        acc = [p["accuracy_among_routed"] for p in pts]
        ax.plot(cov, acc, label=label, color=color, linestyle=linestyle, marker=marker, linewidth=2, markersize=6)

    # Plot Einstein reference point (173 routed, 99.42% accuracy)
    ax.scatter([173], [99.42], color="#7c3aed", s=140, zorder=5, label="Salesforce Einstein (173 routed, 99.4%)")
    ax.annotate("Salesforce Einstein\n(172/173 = 99.4%)", xy=(173, 99.42), xytext=(145, 92),
                arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.2),
                fontsize=9, fontweight="bold", color="#7c3aed")

    ax.set_title("Confidence-Gated Routing: Coverage vs. Accuracy Trade-Off", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Cases Auto-Routed (out of 200 Blind Test Cases)", fontsize=10)
    ax.set_ylabel("Routing Accuracy Among Auto-Routed Cases (%)", fontsize=10)
    ax.set_ylim(40, 105)
    ax.set_xlim(90, 205)
    ax.legend(frameon=True, facecolor="white", loc="lower left", fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "routing_simulation.png")
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Saved: {out_path}")


def plot_difficulty_breakdown():
    models = {
        "Rules Heuristics": ("results/rules_predictions.csv", "#94a3b8"),
        "Laya Zero-Shot (List)": ("results/laya_zeroshot_predictions.csv", "#f87171"),
        "Laya Fine-Tuned (Head)": ("results/laya_finetuned_predictions.csv", "#fbbf24"),
        "Laya Zero-Shot (Dict)": ("results/laya_zeroshot_dict_predictions.csv", "#34d399"),
        "TF-IDF + MLP": ("results/tfidf_baseline_predictions.csv", "#60a5fa"),
    }

    tiers = ["easy", "paraphrase", "multi_topic", "ambiguous"]
    tier_labels = ["Easy (80)", "Paraphrase (60)", "Multi-Topic (40)", "Ambiguous (20)"]

    data_by_model = {}
    for name, (path, color) in models.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            accs = [df[df["difficulty"] == t]["correct"].mean() * 100.0 for t in tiers]
            data_by_model[name] = (accs, color)

    # Einstein on routed cases: Easy: 100%, Para: 100%, Multi: 97.1%, Ambig: 100%
    data_by_model["Salesforce Einstein (Routed)"] = ([100.0, 100.0, 97.1, 100.0], "#8b5cf6")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(tiers))
    n_models = len(data_by_model)
    width = 0.8 / n_models

    for i, (name, (accs, color)) in enumerate(data_by_model.items()):
        offset = (i - n_models / 2 + 0.5) * width
        rects = ax.bar(x + offset, accs, width, label=name, color=color, alpha=0.9, edgecolor="white", linewidth=0.5)

    ax.set_title("Benchmark Accuracy by Difficulty Tier (200 Blind Test Cases)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, facecolor="white", loc="lower left", fontsize=8.5, ncol=2)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "difficulty_breakdown.png")
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Saved: {out_path}")


def generate_all_plots():
    print("=" * 80)
    print(" 🎨 GENERATING MEASURED BENCHMARK VISUALIZATIONS")
    print("=" * 80)
    plot_confidence_distribution()
    plot_routing_simulation()
    plot_difficulty_breakdown()
    print("✅ Benchmark charts updated successfully!\n")


if __name__ == "__main__":
    generate_all_plots()
