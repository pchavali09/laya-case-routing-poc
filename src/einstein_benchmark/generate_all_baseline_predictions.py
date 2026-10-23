"""
Generate All Baseline Predictions & Per-Case CSVs
=================================================
Regenerates and commits per-case prediction CSVs for:
1. Untuned Legacy Rules (src/rules.py) on Original and Hard Test Sets
2. Trained Keyword Rules (src/rules_trained.py, frozen commit ff8b315) on Original and Hard Test Sets
3. TF-IDF + MLP Baseline (Option A: trained on data/einstein_hard_benchmark/train_closed.csv)
4. TF-IDF + MLP Baseline (trained on data/einstein_benchmark/train_closed.csv, 50 epochs and 40 epochs)

Also documents the 198 vs. 197 accuracy difference on the original benchmark:
At 40 epochs (learning rate 1e-3, AdamW), ENS-TEST-0132 is predicted as 'Technical' (197/200).
By 50 epochs, the network converges further and classifies ENS-TEST-0132 as 'Other' (198/200).
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.rules import classify_case_rules as classify_case_legacy
from src.rules_trained import classify_case_trained_rules

DEPARTMENTS = ["Billing", "Technical", "Sales", "Operations", "Other"]
DEPT2IDX = {d: i for i, d in enumerate(DEPARTMENTS)}
IDX2DEPT = {i: d for i, d in enumerate(DEPARTMENTS)}


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=5):
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


def wilson_score_interval(successes: int, total: int):
    if total == 0:
        return 0.0, 0.0, 0.0
    z = 1.959963984540054
    p_hat = successes / total
    denom = 1 + (z**2) / total
    center = (p_hat + (z**2) / (2 * total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * total)) / total) / denom
    return p_hat, max(0.0, center - spread), min(1.0, center + spread)


def evaluate_rules(test_df, mode="trained"):
    results = []
    for _, row in test_df.iterrows():
        if mode == "trained":
            pred_dept, conf, rule_id = classify_case_trained_rules(row["Subject"], row["Description"])
        else:
            pred_dept, conf, rule_id = classify_case_legacy(row["Subject"], row["Description"])
        results.append({
            "ExternalId__c": row["ExternalId__c"],
            "true_department": row["true_department"],
            "predicted_department": pred_dept,
            "confidence": float(conf) if conf is not None else 0.5,
            "rule_id": rule_id,
            "is_correct": bool(pred_dept == row["true_department"])
        })
    return pd.DataFrame(results)


def train_eval_tfidf(train_df, test_df, epochs=50, seed=2026):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_texts = (train_df["Subject"].fillna("") + " " + train_df["Description"].fillna("")).tolist()
    train_labels = [DEPT2IDX[d] for d in train_df["Department__c"]]
    test_texts = (test_df["Subject"].fillna("") + " " + test_df["Description"].fillna("")).tolist()

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
    for _ in range(epochs):
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


def main():
    results_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    orig_dir = os.path.join(REPO_ROOT, "data", "einstein_benchmark")
    hard_dir = os.path.join(REPO_ROOT, "data", "einstein_hard_benchmark")

    # Load datasets
    orig_train_df = pd.read_csv(os.path.join(orig_dir, "train_closed.csv"))
    hard_train_df = pd.read_csv(os.path.join(hard_dir, "train_closed.csv"))

    orig_test = pd.merge(
        pd.read_csv(os.path.join(orig_dir, "test_blind.csv")),
        pd.read_csv(os.path.join(orig_dir, "test_truth.csv")),
        on="ExternalId__c"
    )
    hard_test = pd.merge(
        pd.read_csv(os.path.join(hard_dir, "test_blind.csv")),
        pd.read_csv(os.path.join(hard_dir, "test_truth.csv")),
        on="ExternalId__c"
    )

    print("Generating baseline predictions...")

    # 1. Untuned Legacy Rules
    legacy_orig = evaluate_rules(orig_test, mode="legacy")
    legacy_orig.to_csv(os.path.join(results_dir, "predictions_rules_legacy_orig.csv"), index=False)
    legacy_hard = evaluate_rules(hard_test, mode="legacy")
    legacy_hard.to_csv(os.path.join(results_dir, "predictions_rules_legacy_hard.csv"), index=False)
    print(f"Untuned Legacy Rules: Orig={legacy_orig['is_correct'].sum()}/200, Hard={legacy_hard['is_correct'].sum()}/200")

    # 2. Trained Keyword Rules (Hard 800, commit ff8b315)
    tr_rules_orig = evaluate_rules(orig_test, mode="trained")
    tr_rules_orig.to_csv(os.path.join(results_dir, "predictions_rules_trained_orig.csv"), index=False)
    tr_rules_hard = evaluate_rules(hard_test, mode="trained")
    tr_rules_hard.to_csv(os.path.join(results_dir, "predictions_rules_trained_hard.csv"), index=False)
    print(f"Trained Keyword Rules (Hard-800): Orig={tr_rules_orig['is_correct'].sum()}/200, Hard={tr_rules_hard['is_correct'].sum()}/200")

    # 3. TF-IDF + MLP Baseline (Option A: Hard 800 training)
    tfidf_hard_on_orig = train_eval_tfidf(hard_train_df, orig_test, epochs=50)
    tfidf_hard_on_orig.to_csv(os.path.join(results_dir, "predictions_tfidf_orig.csv"), index=False)
    tfidf_hard_on_hard = train_eval_tfidf(hard_train_df, hard_test, epochs=50)
    tfidf_hard_on_hard.to_csv(os.path.join(results_dir, "predictions_tfidf_hard.csv"), index=False)
    print(f"TF-IDF (Hard-800 Train, 50 epochs): Orig={tfidf_hard_on_orig['is_correct'].sum()}/200, Hard={tfidf_hard_on_hard['is_correct'].sum()}/200")

    # 4. TF-IDF + MLP Baseline (Original 800 training: 50 epochs vs. 40 epochs)
    tfidf_orig_50 = train_eval_tfidf(orig_train_df, orig_test, epochs=50)
    tfidf_orig_50.to_csv(os.path.join(results_dir, "predictions_tfidf_origtrain_origtest_50epochs.csv"), index=False)
    tfidf_orig_40 = train_eval_tfidf(orig_train_df, orig_test, epochs=40)
    tfidf_orig_40.to_csv(os.path.join(results_dir, "predictions_tfidf_origtrain_origtest_40epochs.csv"), index=False)
    tfidf_orig_on_hard = train_eval_tfidf(orig_train_df, hard_test, epochs=50)
    tfidf_orig_on_hard.to_csv(os.path.join(results_dir, "predictions_tfidf_origtrain_hardtest_50epochs.csv"), index=False)

    print(f"TF-IDF (Original-800 Train on Orig Test): 40 epochs={tfidf_orig_40['is_correct'].sum()}/200, 50 epochs={tfidf_orig_50['is_correct'].sum()}/200")
    print(f"TF-IDF (Original-800 Train on Hard Test): 50 epochs={tfidf_orig_on_hard['is_correct'].sum()}/200")

    # Analyze 198 vs 197 difference
    diff_mask = tfidf_orig_40["predicted_department"] != tfidf_orig_50["predicted_department"]
    for idx in np.where(diff_mask)[0]:
        row = orig_test.iloc[idx]
        print(f"\n[198 vs 197 Discrepancy Case]:")
        print(f"ExternalId: {row['ExternalId__c']}")
        print(f"Subject: {row['Subject']}")
        print(f"True Department: {row['true_department']}")
        print(f"40 Epochs Prediction: {tfidf_orig_40.iloc[idx]['predicted_department']}")
        print(f"50 Epochs Prediction: {tfidf_orig_50.iloc[idx]['predicted_department']}")


if __name__ == "__main__":
    main()
