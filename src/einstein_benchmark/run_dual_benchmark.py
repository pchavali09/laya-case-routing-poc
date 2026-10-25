"""
Dual Benchmark Runner: Original Templated Set vs. Group-Held-Out Hard Set
========================================================================
Runs and evaluates:
1. TF-IDF + MLP Baseline (Option A: trained on data/einstein_hard_benchmark/train_closed.csv)
2. Trained Keyword Rules Baseline (tuned from training cases)
3. Untuned Legacy Rules Baseline
4. Real Laya English Zero-Shot (List Criteria)
5. Real Laya English Zero-Shot (Dict Criteria - clean, training-derived)
6. Real Laya Multilingual Zero-Shot (List Criteria)
7. Real Laya Multilingual Zero-Shot (Dict Criteria - clean, training-derived)
8. Salesforce Einstein (Original Set measured; Hard Set from sandbox export or Pending)

Computes for both datasets:
- Total Cases, Cases Routed, Coverage %
- Accuracy among Routed Cases (with Wilson 95% CI)
- Overall Top-1 Accuracy (with Wilson 95% CI)
- Saves per-case prediction CSVs with latency, artifact ID, model name to results/
"""

import os
import sys
import json
import time
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer

# Cap PyTorch CPU threads for system responsiveness
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.rules import classify_case_rules
from src.rules_trained import classify_case_trained_rules
import laya

DEPARTMENTS = ["Billing", "Technical", "Sales", "Operations", "Other"]
DEPT2IDX = {d: i for i, d in enumerate(DEPARTMENTS)}
IDX2DEPT = {i: d for i, d in enumerate(DEPARTMENTS)}

# Descriptions written strictly from training themes; zero wording derived from hard-set themes.
DICT_CRITERIA = {
    "Billing": "invoices, payments, refunds, charges, subscriptions, pricing",
    "Technical": "system errors, software bugs, outages, API integration, technical issues",
    "Sales": "quotes, enterprise licensing, contract renewals, upgrades, sales demos",
    "Operations": "shipping, logistics, warehouse, facilities, hardware procurement",
    "Other": "general inquiries, uncategorized corporate communications, miscellaneous feedback"
}

def compute_ece(confidences, accuracies, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper if i < n_bins - 1 else confidences <= bin_upper)
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)
    return float(ece)

def wilson_score_interval(successes, total, confidence=0.95):
    if total == 0:
        return 0.0, 0.0, 0.0
    z = 1.959963984540054  # 95% confidence
    p_hat = successes / total
    denom = 1 + (z**2) / total
    center = (p_hat + (z**2) / (2 * total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * total)) / total) / denom
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return p_hat, lower, upper


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.net(x)
        return logits / torch.clamp(self.temperature, min=0.1)


def train_tfidf_mlp(train_df, test_df, seed=2026):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_texts = (train_df["Subject"] + " " + train_df["Description"]).tolist()
    train_labels = [DEPT2IDX[d] for d in train_df["Department__c"]]
    test_texts = (test_df["Subject"] + " " + test_df["Description"]).tolist()

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    model = MLPClassifier(X_train.shape[1], hidden_dim=64, num_classes=5)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(train_labels, dtype=torch.long)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)

    model.train()
    for _ in range(50):
        optimizer.zero_grad()
        out = model(X_tr_t)
        loss = criterion(out, y_tr_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_te_t)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)

    results = []
    for i, row in test_df.reset_index(drop=True).iterrows():
        pred_label = IDX2DEPT[preds[i]]
        results.append({
            "ExternalId__c": row["ExternalId__c"],
            "true_department": row["true_department"],
            "predicted_department": pred_label,
            "confidence": float(confs[i]),
            "is_correct": bool(pred_label == row["true_department"])
        })
    return pd.DataFrame(results)


def run_rules_baseline(test_df):
    results = []
    for _, row in test_df.iterrows():
        pred_dept, conf, rule_id = classify_case_rules(row["Subject"], row["Description"])
        results.append({
            "ExternalId__c": row["ExternalId__c"],
            "true_department": row["true_department"],
            "predicted_department": pred_dept,
            "confidence": float(conf) if conf is not None else 0.5,
            "is_correct": bool(pred_dept == row["true_department"])
        })
    return pd.DataFrame(results)


def run_trained_rules_baseline(test_df):
    results = []
    for _, row in test_df.iterrows():
        pred_dept, conf, rule_id = classify_case_trained_rules(row["Subject"], row["Description"])
        results.append({
            "ExternalId__c": row["ExternalId__c"],
            "true_department": row["true_department"],
            "predicted_department": pred_dept,
            "confidence": float(conf),
            "is_correct": bool(pred_dept == row["true_department"])
        })
    return pd.DataFrame(results)


def run_laya_eval(test_df, dataset_key, model_alias="english", criteria_mode="dict"):
    model_checkpoint = "convaiinnovations/laya" if model_alias == "english" else "convaiinnovations/laya-multilingual"
    subfolder = None if model_alias == "english" else "multilingual"
    print(f" -> Evaluating Laya ({model_alias} [{model_checkpoint}], criteria={criteria_mode})...")
    
    agent = laya.load("convaiinnovations/laya", subfolder=subfolder)
    criteria = DEPARTMENTS if criteria_mode == "list" else DICT_CRITERIA
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this customer support case?",
            "criteria": criteria
        }
    }
    artifact_id = f"laya_{model_alias}_{criteria_mode}"

    results = []
    t_start = time.time()
    for _, row in test_df.iterrows():
        state = f"Subject: {row['Subject']}\nDescription: {row['Description']}"
        case_t0 = time.time()
        res = agent.predict(state, questions)
        case_lat = (time.time() - case_t0) * 1000.0
        ans = res["answers"]["department"]
        pred = ans["choice"]
        conf = float(ans.get("confidence", ans.get("answer_confidence", 0.0)))
        results.append({
            "ExternalId__c": row["ExternalId__c"],
            "true_department": row["true_department"],
            "predicted_department": pred,
            "confidence": conf,
            "latency_ms": round(case_lat, 2),
            "model_checkpoint": model_checkpoint,
            "artifact_identifier": artifact_id,
            "criteria_type": criteria_mode,
            "is_correct": bool(pred == row["true_department"])
        })
    elapsed = time.time() - t_start
    print(f"    Completed in {elapsed:.1f}s ({elapsed/len(test_df)*1000:.1f} ms/case)")
    pred_df = pd.DataFrame(results)

    # Save committed per-case prediction CSV
    results_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = f"predictions_zeroshot_{dataset_key}_{model_alias}_{criteria_mode}.csv"
    pred_df.to_csv(os.path.join(results_dir, csv_filename), index=False)
    return pred_df


def score_system_predictions(pred_df, target_coverage_cases=173, model_label="Model", training_label="no training"):
    total = len(pred_df)
    overall_correct = int(pred_df["is_correct"].sum())
    p_ovr, low_ovr, up_ovr = wilson_score_interval(overall_correct, total)

    sorted_df = pred_df.sort_values(by="confidence", ascending=False).reset_index(drop=True)
    routed_df = sorted_df.iloc[:target_coverage_cases]

    routed_count = len(routed_df)
    routed_correct = int(routed_df["is_correct"].sum())
    p_rout, low_rout, up_rout = wilson_score_interval(routed_correct, routed_count)

    ece = None
    if "confidence" in pred_df.columns:
        confs = pred_df["confidence"].values
        accs = pred_df["is_correct"].values.astype(float)
        ece = round(compute_ece(confs, accs), 4)

    return {
        "model": model_label,
        "training": training_label,
        "total_cases": total,
        "overall_correct": overall_correct,
        "overall_accuracy": p_ovr,
        "overall_ci": [low_ovr, up_ovr],
        "cases_routed": routed_count,
        "coverage_pct": routed_count / total,
        "routed_correct": routed_correct,
        "routed_accuracy": p_rout,
        "routed_ci": [low_rout, up_rout],
        "ece": ece
    }


def evaluate_dataset(dataset_name, dataset_key, data_dir, train_csv_path, einstein_preds_path=None):
    print(f"\n================================================================================")
    print(f"EVALUATING DATASET: {dataset_name} ({data_dir})")
    print(f"Training Baseline Data Source: {train_csv_path}")
    print(f"================================================================================")

    train_df = pd.read_csv(train_csv_path)
    test_blind = pd.read_csv(os.path.join(data_dir, "test_blind.csv"))
    test_truth = pd.read_csv(os.path.join(data_dir, "test_truth.csv"))
    test_df = pd.merge(test_blind, test_truth, on="ExternalId__c")

    summary_list = []

    # 1. Untuned Legacy Rules Baseline
    print(" -> Evaluating Untuned Legacy Rules Baseline (src/rules.py)...")
    rules_preds = run_rules_baseline(test_df)
    rules_preds["model_checkpoint"] = "heuristic_regex"
    rules_preds["artifact_identifier"] = f"rules_legacy_{dataset_key}"
    rules_preds["criteria_type"] = "regex"
    rules_preds.to_csv(os.path.join(REPO_ROOT, "results", f"predictions_rules_legacy_{dataset_key}.csv"), index=False)
    summary_list.append(score_system_predictions(rules_preds, 173, "Untuned Legacy Rules Baseline", "no training"))

    # 2. Trained Keyword Rules Baseline (tuned on training cases)
    print(" -> Evaluating Trained Keyword Rules Baseline (src/rules_trained.py)...")
    trained_rules_preds = run_trained_rules_baseline(test_df)
    trained_rules_preds["model_checkpoint"] = "src/rules_trained.py (commit ff8b315)"
    trained_rules_preds["artifact_identifier"] = f"rules_trained_{dataset_key}"
    trained_rules_preds["criteria_type"] = "keyword_rules"
    trained_rules_preds.to_csv(os.path.join(REPO_ROOT, "results", f"predictions_rules_trained_{dataset_key}.csv"), index=False)
    summary_list.append(score_system_predictions(trained_rules_preds, 173, "Trained Keyword Rules Baseline", "trained on 800 closed cases"))

    # 3. TF-IDF + MLP (Control Baseline)
    print(" -> Training & Evaluating TF-IDF + MLP (Control Baseline)...")
    tfidf_preds = train_tfidf_mlp(train_df, test_df)
    tfidf_preds["model_checkpoint"] = "tfidf_mlp_seed2026_50epochs"
    tfidf_preds["artifact_identifier"] = f"tfidf_mlp_{dataset_key}"
    tfidf_preds["criteria_type"] = "supervised_mlp"
    tfidf_preds.to_csv(os.path.join(REPO_ROOT, "results", f"predictions_tfidf_{dataset_key}.csv"), index=False)
    summary_list.append(score_system_predictions(tfidf_preds, 173, "TF-IDF + MLP (Control Baseline)", "trained on 800 closed cases"))

    # 4. Laya English Zero-Shot (Dict Criteria - clean training-derived)
    laya_en_dict_preds = run_laya_eval(test_df, dataset_key, model_alias="english", criteria_mode="dict")
    summary_list.append(score_system_predictions(laya_en_dict_preds, 173, "Laya English Zero-Shot (Dict Criteria)", "no training"))

    # 5. Laya English Zero-Shot (List Criteria)
    laya_en_list_preds = run_laya_eval(test_df, dataset_key, model_alias="english", criteria_mode="list")
    summary_list.append(score_system_predictions(laya_en_list_preds, 173, "Laya English Zero-Shot (List Criteria)", "no training"))

    # 6. Laya Multilingual Zero-Shot (Dict Criteria - clean training-derived)
    laya_multi_dict_preds = run_laya_eval(test_df, dataset_key, model_alias="multilingual", criteria_mode="dict")
    summary_list.append(score_system_predictions(laya_multi_dict_preds, 173, "Laya Multilingual Zero-Shot (Dict Criteria)", "no training"))

    # 7. Laya Multilingual Zero-Shot (List Criteria)
    laya_multi_list_preds = run_laya_eval(test_df, dataset_key, model_alias="multilingual", criteria_mode="list")
    summary_list.append(score_system_predictions(laya_multi_list_preds, 173, "Laya Multilingual Zero-Shot (List Criteria)", "no training"))

    # Check differences between English and Multilingual predictions
    diff_list = int((laya_en_list_preds["predicted_department"] != laya_multi_list_preds["predicted_department"]).sum())
    diff_dict = int((laya_en_dict_preds["predicted_department"] != laya_multi_dict_preds["predicted_department"]).sum())
    print(f"\n   [Divergence Check ({dataset_name})]")
    print(f"   - List Criteria: English vs Multilingual differ on {diff_list}/{len(test_df)} cases ({diff_list/len(test_df)*100:.1f}%)")
    print(f"   - Dict Criteria: English vs Multilingual differ on {diff_dict}/{len(test_df)} cases ({diff_dict/len(test_df)*100:.1f}%)")

    # 8. Salesforce Einstein (if provided)
    if einstein_preds_path and os.path.exists(einstein_preds_path):
        ein_df = pd.read_csv(einstein_preds_path)
        m_df = pd.merge(test_df, ein_df, on="ExternalId__c")

        pred_col = "Einstein_Department" if "Einstein_Department" in m_df.columns else "predicted_department"
        if "is_routed" in m_df.columns:
            routed_mask = m_df["is_routed"].astype(str).str.lower() == "true"
        else:
            routed_mask = m_df[pred_col] != "Unrouted"

        m_df["is_correct"] = (m_df[pred_col] == m_df["true_department"]) & routed_mask
        tot = len(m_df)
        ovr_corr = int(m_df["is_correct"].sum())
        p_ovr, l_ovr, u_ovr = wilson_score_interval(ovr_corr, tot)

        routed_ein = m_df[routed_mask]
        rout_count = len(routed_ein)
        rout_corr = int((routed_ein[pred_col] == routed_ein["true_department"]).sum())
        p_rout, l_rout, u_rout = wilson_score_interval(rout_corr, rout_count)

        conf_col = "Einstein_Confidence" if "Einstein_Confidence" in m_df.columns else "confidence"
        ein_ece = None
        if conf_col in m_df.columns:
            ein_ece = round(compute_ece(m_df[conf_col].values, m_df["is_correct"].values.astype(float)), 4)

        summary_list.append({
            "model": "Salesforce Einstein Case Classification",
            "training": "trained on 800 closed cases",
            "total_cases": tot,
            "overall_correct": ovr_corr,
            "overall_accuracy": p_ovr,
            "overall_ci": [l_ovr, u_ovr],
            "cases_routed": rout_count,
            "coverage_pct": rout_count / tot,
            "routed_correct": rout_corr,
            "routed_accuracy": p_rout,
            "routed_ci": [l_rout, u_rout],
            "ece": ein_ece
        })
    else:
        summary_list.append({
            "model": "Salesforce Einstein Case Classification",
            "training": "trained on 800 closed cases",
            "total_cases": len(test_df),
            "overall_correct": None,
            "overall_accuracy": None,
            "overall_ci": None,
            "cases_routed": None,
            "coverage_pct": None,
            "routed_correct": None,
            "routed_accuracy": None,
            "routed_ci": None,
            "ece": None,
            "status": "Einstein scored in a Salesforce sandbox; see EINSTEIN_HARD_RUN_NOTES.md"
        })

    return summary_list


def main():
    orig_dir = os.path.join(REPO_ROOT, "data", "einstein_benchmark")
    hard_dir = os.path.join(REPO_ROOT, "data", "einstein_hard_benchmark")

    # Option A: unified training set is data/einstein_hard_benchmark/train_closed.csv everywhere
    unified_train_csv = os.path.join(hard_dir, "train_closed.csv")

    orig_einstein_csv = os.path.join(orig_dir, "einstein_predictions.csv")
    hard_einstein_csv = os.path.join(hard_dir, "einstein_predictions.csv")

    archive_dir = os.path.join(REPO_ROOT, "results", "archive_original_test")
    os.makedirs(archive_dir, exist_ok=True)
    out_orig_json = os.path.join(archive_dir, "measured_benchmark_original_summary.json")
    out_hard_json = os.path.join(REPO_ROOT, "results", "measured_benchmark_hard_summary.json")

    orig_summary = evaluate_dataset("Original Templated Benchmark", "orig", orig_dir, unified_train_csv, orig_einstein_csv)
    with open(out_orig_json, "w") as f:
        json.dump(orig_summary, f, indent=2)

    hard_summary = evaluate_dataset("Group-Held-Out Hard Benchmark", "hard", hard_dir, unified_train_csv, hard_einstein_csv)
    with open(out_hard_json, "w") as f:
        json.dump(hard_summary, f, indent=2)

    print("\n✅ Benchmark evaluations complete. Summary files and per-case prediction CSVs written to results/.")

if __name__ == "__main__":
    main()
