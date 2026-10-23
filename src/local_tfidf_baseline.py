"""
Simple Local Baseline: TF-IDF + Calibrated MLP
-----------------------------------------------
This is a clearly labelled classical NLP baseline using:
- Scikit-learn TfidfVectorizer (5000 n-gram features, unigram + bigram)
- PyTorch 2-layer MLP (hidden_dim=64, ReLU, Dropout=0.2)
- Temperature calibration scaling

NOTE: This is a simple local baseline, NOT the Laya foundation model.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPARTMENTS = ["Billing", "Technical", "Sales", "Operations", "Other"]
DEPT2IDX = {dept: i for i, dept in enumerate(DEPARTMENTS)}
IDX2DEPT = {i: dept for i, dept in enumerate(DEPARTMENTS)}


class SimpleMLP(nn.Module):
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
            scaled = logits / torch.clamp(self.temperature, min=0.1, max=10.0)
            probs = torch.softmax(scaled, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)
            return preds, confidences, probs


def train_and_eval_tfidf_baseline():
    train_path = os.path.join(BASE_DIR, "data", "einstein_hard_benchmark", "train_closed.csv")
    test_blind_path = os.path.join(BASE_DIR, "data", "einstein_hard_benchmark", "test_blind.csv")
    test_truth_path = os.path.join(BASE_DIR, "data", "einstein_hard_benchmark", "test_truth.csv")

    train_df = pd.read_csv(train_path)
    test_blind = pd.read_csv(test_blind_path)
    test_truth = pd.read_csv(test_truth_path)
    test_df = pd.merge(test_blind, test_truth, on="ExternalId__c")

    train_texts = (train_df["Subject"] + " " + train_df["Description"]).tolist()
    train_labels = [DEPT2IDX[d] for d in train_df["Department__c"]]
    test_texts = (test_df["Subject"] + " " + test_df["Description"]).tolist()
    test_labels = [DEPT2IDX[d] for d in test_df["true_department"]]

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    torch.manual_seed(2026)
    np.random.seed(2026)

    model = SimpleMLP(X_train.shape[1], hidden_dim=64, num_classes=5)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(train_labels, dtype=torch.long)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)

    model.train()
    for _ in range(40):
        optimizer.zero_grad()
        loss = criterion(model(X_tr_t), y_tr_t)
        loss.backward()
        optimizer.step()

    model.eval()
    preds, confs, probs = model.calibrated_predict_proba(X_te_t)
    preds_np = preds.numpy()
    correct = (preds_np == np.array(test_labels)).sum()
    acc = correct / len(test_labels)

    print(f"📊 [Simple Local Baseline: TF-IDF + MLP] Test Accuracy: {correct}/{len(test_labels)} ({acc*100:.2f}%)")
    return model, vectorizer, acc


if __name__ == "__main__":
    train_and_eval_tfidf_baseline()
