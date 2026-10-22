"""
Laya Model Training & Evaluation on Einstein 1,000-Case Benchmark
-----------------------------------------------------------------
Phase 2 - Step 4:
1. Train Laya classifier head with temperature calibration on train_closed.csv (800 cases, seed 2026).
2. Evaluate Rules baseline on test_truth.csv.
3. Report overall accuracy and accuracy by difficulty (easy, paraphrase, multi_topic, ambiguous).
4. Compute confidence-gate table for Laya across tau in [0.50, 0.95].
5. Export predictions to data/einstein_benchmark/laya_predictions.csv (ExternalId__c, predicted_department, confidence).
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Import Rules baseline from src/rules.py
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.rules import classify_case_rules

DEPARTMENTS = ["Billing", "Technical", "Sales", "Operations", "Other"]
DEPT2IDX = {dept: i for i, dept in enumerate(DEPARTMENTS)}
IDX2DEPT = {i: dept for i, dept in enumerate(DEPARTMENTS)}
DIFFICULTIES = ["easy", "paraphrase", "multi_topic", "ambiguous"]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


class LayaClassifierHead(nn.Module):
    """
    Calibrated neural classifier head for domain-specific routing.
    Matches the architecture in src/laya_finetune.py.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def calibrated_predict_proba(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits = self.forward(x)
            scaled_logits = logits / torch.clamp(self.temperature, min=0.1, max=10.0)
            probs = torch.softmax(scaled_logits, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)
            return preds, confidences, probs


def set_seed(seed: int = 2026):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_benchmark():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(base_dir, "data", "einstein_benchmark")
    
    train_csv_path = os.path.join(data_dir, "train_closed.csv")
    test_blind_path = os.path.join(data_dir, "test_blind.csv")
    test_truth_path = os.path.join(data_dir, "test_truth.csv")
    out_preds_path = os.path.join(data_dir, "laya_predictions.csv")

    set_seed(2026)

    print("=" * 65)
    print(" 🚀 PHASE 2 - STEP 4: LAYA TRAINING & BENCHMARK EVALUATION")
    print("=" * 65)

    # 1. Load data
    train_df = pd.read_csv(train_csv_path)
    test_blind_df = pd.read_csv(test_blind_path)
    test_truth_df = pd.read_csv(test_truth_path)

    # Merge test data to get Subject, Description, true_department, and difficulty
    test_df = pd.merge(test_blind_df, test_truth_df, on="ExternalId__c")
    
    train_texts = (train_df["Subject"].fillna("") + " " + train_df["Description"].fillna("")).tolist()
    test_texts = (test_df["Subject"].fillna("") + " " + test_df["Description"].fillna("")).tolist()
    
    y_train_raw = [DEPT2IDX[d] for d in train_df["Department__c"]]
    y_test_raw = [DEPT2IDX[d] for d in test_df["true_department"]]

    # Train / Val split for calibration (80/20 stratified split, seed 2026)
    train_idx, val_idx = train_test_split(
        range(len(train_texts)),
        test_size=0.20,
        random_state=2026,
        stratify=y_train_raw
    )

    train_sub_texts = [train_texts[i] for i in train_idx]
    val_sub_texts = [train_texts[i] for i in val_idx]
    y_train_sub = [y_train_raw[i] for i in train_idx]
    y_val_sub = [y_train_raw[i] for i in val_idx]

    # Feature extraction (TF-IDF sublinear)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=512,
        sublinear_tf=True
    )
    X_train_sub = vectorizer.fit_transform(train_sub_texts).toarray()
    X_val_sub = vectorizer.transform(val_sub_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    X_train_t = torch.tensor(X_train_sub, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_sub, dtype=torch.long)
    X_val_t = torch.tensor(X_val_sub, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_sub, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_raw, dtype=torch.long)

    # 2. Train Laya neural head
    model = LayaClassifierHead(input_dim=X_train_sub.shape[1], hidden_dim=64, num_classes=len(DEPARTMENTS))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.net.parameters(), lr=0.015, weight_decay=1e-4)

    epochs = 80
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

    # 3. Fit Temperature Calibration on validation set
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)

    temp_optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=50)
    def eval_temp():
        temp_optimizer.zero_grad()
        loss = criterion(val_logits / torch.clamp(model.temperature, min=0.1), y_val_t)
        loss.backward()
        return loss
    temp_optimizer.step(eval_temp)
    calibrated_temp = float(model.temperature.item())
    print(f"🔥 Calibrated Temperature Parameter: T = {calibrated_temp:.4f}")

    # 4. Predict on Test Set
    preds_t, confs_t, probs_t = model.calibrated_predict_proba(X_test_t)
    preds = preds_t.numpy()
    confs = confs_t.numpy()
    y_test_arr = y_test_t.numpy()

    # Save Laya Predictions
    laya_preds_df = pd.DataFrame({
        "ExternalId__c": test_df["ExternalId__c"],
        "predicted_department": [IDX2DEPT[p] for p in preds],
        "confidence": [round(float(c), 4) for c in confs]
    })
    laya_preds_df.to_csv(out_preds_path, index=False)
    print(f"✅ Saved Laya predictions to: {out_preds_path}")

    # 5. Evaluate Rules Baseline
    rules_preds = []
    for _, row in test_df.iterrows():
        r_pred, _, _ = classify_case_rules(str(row["Subject"]), str(row["Description"]))
        rules_preds.append(r_pred)
    rules_preds_arr = np.array([DEPT2IDX[p] for p in rules_preds])

    # 6. Overall Accuracies
    laya_correct = (preds == y_test_arr)
    rules_correct = (rules_preds_arr == y_test_arr)

    laya_acc = laya_correct.mean()
    rules_acc = rules_correct.mean()

    # Difficulty Breakdowns
    print("\n" + "=" * 65)
    print(" 📊 1. OVERALL ACCURACY & ACCURACY BY DIFFICULTY")
    print("=" * 65)
    print(f"{'Tier':<14} | {'Count':<6} | {'Rules Baseline':<16} | {'Laya Classifier':<16}")
    print("-" * 65)
    print(f"{'OVERALL':<14} | {len(test_df):<6} | {rules_correct.sum()}/{len(test_df)} ({rules_acc:.1%}){' ':<6} | {laya_correct.sum()}/{len(test_df)} ({laya_acc:.1%})")
    print("-" * 65)

    difficulty_stats = {}
    for diff in DIFFICULTIES:
        mask = (test_df["difficulty"] == diff).values
        n_diff = mask.sum()
        r_corr = rules_correct[mask].sum()
        l_corr = laya_correct[mask].sum()
        r_pct = r_corr / n_diff if n_diff > 0 else 0
        l_pct = l_corr / n_diff if n_diff > 0 else 0
        difficulty_stats[diff] = {
            "count": int(n_diff),
            "rules_correct": int(r_corr),
            "rules_acc": float(r_pct),
            "laya_correct": int(l_corr),
            "laya_acc": float(l_pct)
        }
        print(f"{diff:<14} | {n_diff:<6} | {r_corr}/{n_diff} ({r_pct:.1%}){' ':<8} | {l_corr}/{n_diff} ({l_pct:.1%})")
    print("=" * 65)

    # 7. Confidence Gate Table for Laya
    print("\n" + "=" * 65)
    print(" 🎯 2. LAYA CONFIDENCE GATING TABLE (tau in [0.50, 0.95])")
    print("=" * 65)
    print(f"{'Threshold (tau)':<16} | {'Auto-Routed %':<14} | {'Auto-Routed Acc':<16} | {'Human Routed (Count)':<20}")
    print("-" * 72)

    gate_table = []
    for tau in THRESHOLDS:
        routed_mask = (confs >= tau)
        n_routed = routed_mask.sum()
        pct_routed = (n_routed / len(test_df)) * 100.0
        n_human = len(test_df) - n_routed
        
        if n_routed > 0:
            routed_acc = (preds[routed_mask] == y_test_arr[routed_mask]).mean() * 100.0
            acc_str = f"{routed_acc:.1f}%"
        else:
            routed_acc = None
            acc_str = "N/A (0 cases)"

        gate_table.append({
            "tau": tau,
            "auto_routed_pct": round(pct_routed, 1),
            "auto_routed_acc": round(routed_acc, 1) if routed_acc is not None else None,
            "human_routed_count": int(n_human),
            "auto_routed_count": int(n_routed)
        })
        print(f"{tau:<16.2f} | {pct_routed:>6.1f}%{' ':<7} | {acc_str:<16} | {n_human:<20}")
    print("=" * 72)

    # Save summary report for logging & artifact generation
    summary_report = {
        "model": "Laya Classifier (Temperature-Calibrated)",
        "seed": 2026,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "overall_rules_acc": float(rules_acc),
        "overall_laya_acc": float(laya_acc),
        "difficulty_breakdown": difficulty_stats,
        "confidence_gate_table": gate_table
    }
    with open(os.path.join(data_dir, "laya_benchmark_summary.json"), "w") as f:
        json.dump(summary_report, f, indent=2)

    return summary_report


if __name__ == "__main__":
    run_benchmark()
