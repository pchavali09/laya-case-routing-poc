"""
Laya Fine-Tuning & Temperature Calibration
-------------------------------------------
Adapts Laya's decision engine to domain-specific enterprise support cases using:
1. Supervised domain adaptation on the 140 training cases.
2. Temperature calibration (Platt scaling / temperature optimization) on 30 validation cases.
3. Out-of-sample evaluation on 30 test cases.
4. Export of calibrated model weights to results/laya_finetuned.pt & laya_calibration.json.
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

DEPARTMENTS = ["Billing", "Technical", "Sales", "Operations", "Other"]
DEPT2IDX = {dept: i for i, dept in enumerate(DEPARTMENTS)}
IDX2DEPT = {i: dept for i, dept in enumerate(DEPARTMENTS)}


class LayaClassifierHead(nn.Module):
    """
    Lightweight, high-speed neural decision head with learnable temperature calibration.
    Designed for local-first System 1 latency (<15ms per inference).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        # Learnable inverse temperature parameter for post-hoc calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits

    def calibrated_predict_proba(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits = self.forward(x)
            # Temperature scaling: logits / T
            scaled_logits = logits / torch.clamp(self.temperature, min=0.1, max=10.0)
            probs = torch.softmax(scaled_logits, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)
            return preds, confidences, probs


def prepare_features(train_texts, val_texts, test_texts):
    """Extracts rich n-gram feature representations."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=512,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_val = vectorizer.transform(val_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()
    return vectorizer, X_train, X_val, X_test


def train_laya_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    splits_dir = os.path.join(base_dir, "data", "splits")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("==================================================")
    print(" ⚡ FINE-TUNING LAYA CLASSIFIER ON 140 CASES      ")
    print("==================================================")

    # Load splits
    train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(splits_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))

    train_texts = (train_df["subject"] + " " + train_df["body"]).tolist()
    val_texts = (val_df["subject"] + " " + val_df["body"]).tolist()
    test_texts = (test_df["subject"] + " " + test_df["body"]).tolist()

    y_train = torch.tensor([DEPT2IDX[d] for d in train_df["department"]], dtype=torch.long)
    y_val = torch.tensor([DEPT2IDX[d] for d in val_df["department"]], dtype=torch.long)
    y_test = torch.tensor([DEPT2IDX[d] for d in test_df["department"]], dtype=torch.long)

    vectorizer, X_train, X_val, X_test = prepare_features(train_texts, val_texts, test_texts)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # Deterministic training seed
    torch.manual_seed(42)
    np.random.seed(42)

    model = LayaClassifierHead(input_dim=X_train.shape[1], hidden_dim=64, num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.net.parameters(), lr=0.015, weight_decay=1e-4)

    # Train loop
    epochs = 60
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    # Step 2: Temperature calibration on Validation set (Platt scaling)
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
    
    # Optimize temperature T to minimize cross-entropy on validation set
    temp_optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=50)
    def eval_temp():
        temp_optimizer.zero_grad()
        loss = criterion(val_logits / torch.clamp(model.temperature, min=0.1), y_val)
        loss.backward()
        return loss
    temp_optimizer.step(eval_temp)
    calibrated_temp = float(model.temperature.item())
    print(f"🔥 Fitted Optimal Calibration Temperature: T = {calibrated_temp:.3f}")

    # Step 3: Evaluate on Test set
    preds, confs, probs = model.calibrated_predict_proba(X_test_t)
    preds_np = preds.numpy()
    confs_np = confs.numpy()
    y_test_np = y_test.numpy()

    correct_mask = (preds_np == y_test_np)
    test_accuracy = correct_mask.mean()
    print(f"📊 [Fine-Tuned Laya] Test Accuracy: {correct_mask.sum()}/{len(y_test_np)} ({test_accuracy:.1%})")
    print(f"   Mean Calibrated Confidence: {confs_np.mean():.3f} (Range: [{confs_np.min():.3f}, {confs_np.max():.3f}])")

    # Save artifacts
    model_checkpoint_path = os.path.join(results_dir, "laya_finetuned.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": X_train.shape[1],
        "temperature": calibrated_temp,
        "classes": DEPARTMENTS,
        "vocabulary": vectorizer.vocabulary_,
        "idf": vectorizer.idf_.tolist()
    }, model_checkpoint_path)

    calibration_meta = {
        "temperature": calibrated_temp,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "test_accuracy": float(test_accuracy),
        "mean_confidence": float(confs_np.mean()),
        "classes": DEPARTMENTS
    }
    with open(os.path.join(results_dir, "laya_calibration.json"), "w") as f:
        json.dump(calibration_meta, f, indent=2)

    # Save test predictions
    test_results = []
    for i, row in test_df.iterrows():
        test_results.append({
            "case_id": row["case_id"],
            "subject": row["subject"],
            "ground_truth": row["department"],
            "prediction": IDX2DEPT[preds_np[i]],
            "confidence": round(float(confs_np[i]), 4),
            "correct": bool(correct_mask[i]),
            "calibrated": True,
            "department_probabilities": {
                dept: round(float(probs[i][j]), 4) for j, dept in enumerate(DEPARTMENTS)
            }
        })
    pred_df = pd.DataFrame(test_results)
    pred_df.to_csv(os.path.join(results_dir, "laya_finetuned_predictions.csv"), index=False)

    print(f"✅ Checkpoint saved to: {model_checkpoint_path}")
    print(f"✅ Predictions saved to: results/laya_finetuned_predictions.csv\n")
    return model, vectorizer, test_results


if __name__ == "__main__":
    train_laya_model()
