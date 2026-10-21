"""
Laya Inference & Confidence-Gated Decision Engine
-------------------------------------------------
Loads the fine-tuned and temperature-calibrated model to perform:
1. Low-latency local inference (<10ms).
2. Well-calibrated posterior confidence estimation.
3. Policy-based confidence gating:
   - confidence >= threshold -> AUTO_ROUTE
   - confidence < threshold  -> HUMAN_TRIAGE_ESCALATION
4. Automatic language detection.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

DEPARTMENTS = ["Billing", "Technical", "Sales", "Operations", "Other"]
DEFAULT_THRESHOLD = 0.85


class LayaInferenceEngine:
    def __init__(self, checkpoint_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if checkpoint_path is None:
            checkpoint_path = os.path.join(base_dir, "results", "laya_finetuned.pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Run laya_finetune.py first.")

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
        self.classes = checkpoint["classes"]
        self.temperature = float(checkpoint["temperature"])
        input_dim = checkpoint["input_dim"]

        # Restore vectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=512, sublinear_tf=True)
        self.vectorizer.vocabulary_ = checkpoint["vocabulary"]
        self.vectorizer.idf_ = np.array(checkpoint["idf"])

        # Reconstruct neural head
        from src.laya_finetune import LayaClassifierHead
        self.model = LayaClassifierHead(input_dim=input_dim, hidden_dim=64, num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def detect_case_language(self, text: str) -> str:
        """Lightweight language detector for cross-lingual routing."""
        # Check Devanagari script for Hindi
        if any("\u0900" <= char <= "\u097F" for char in text):
            return "Hindi"
        # Check Spanish accents and inverted punctuation
        if any(char in "¿¡áéíóúüñ" for char in text.lower()):
            return "Spanish"
        # Check CJK for Mandarin
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            return "Mandarin"
        return "English"

    def classify_case(
        self,
        subject: str,
        body: str,
        confidence_threshold: float = DEFAULT_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Runs calibrated inference with uncertainty gating.
        """
        combined_text = f"{subject} {body}"
        detected_lang = self.detect_case_language(combined_text)

        # Feature transform
        X = self.vectorizer.transform([combined_text]).toarray()
        X_t = torch.tensor(X, dtype=torch.float32)

        preds, confs, probs = self.model.calibrated_predict_proba(X_t)
        pred_idx = preds.item()
        confidence = round(float(confs.item()), 4)
        predicted_dept = self.classes[pred_idx]

        # Action policy gating
        if confidence >= confidence_threshold:
            action = "AUTO_ROUTE"
            routing_reason = f"Confidence {confidence:.2%} exceeds auto-route threshold {confidence_threshold:.2%}"
        else:
            action = "HUMAN_TRIAGE_ESCALATION"
            routing_reason = f"Confidence {confidence:.2%} below auto-route threshold {confidence_threshold:.2%} (flagged for review)"

        prob_distribution = {
            dept: round(float(probs[0][i].item()), 4)
            for i, dept in enumerate(self.classes)
        }

        return {
            "predicted_department": predicted_dept,
            "confidence": confidence,
            "calibrated": True,
            "decision": action,
            "routing_reason": routing_reason,
            "language": detected_lang,
            "confidence_threshold": confidence_threshold,
            "probabilities": prob_distribution
        }


if __name__ == "__main__":
    import sys
    engine = LayaInferenceEngine()
    test_subject = "Urgent payment gateway error on renewal"
    test_body = "Our automatic card renewal failed with code 402. Please fix immediately."
    
    if len(sys.argv) > 2:
        test_subject = sys.argv[1]
        test_body = sys.argv[2]

    res = engine.classify_case(test_subject, test_body)
    print("\n🔍 LAYA CLASSIFICATION RESULT:")
    print(json.dumps(res, indent=2))
