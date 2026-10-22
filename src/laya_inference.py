"""
Laya Inference & Confidence-Gated Decision Engine
-------------------------------------------------
Uses real Laya (convaiinnovations/laya via pip `laya`) to perform:
1. Local transformer-based routing decisions using choice questions.
2. Posterior confidence estimation and option probability distribution.
3. Policy-based confidence gating:
   - confidence >= threshold -> AUTO_ROUTE
   - confidence < threshold  -> HUMAN_TRIAGE_ESCALATION
4. Native script and language routing from Laya's router.
"""

import os
import sys
import time
from typing import Dict, Any, Optional

import laya
from laya import Router

DEPARTMENTS = ["Billing", "Technical", "Sales", "Operations", "Other"]
DEFAULT_THRESHOLD = 0.85


class LayaInferenceEngine:
    """
    Production inference engine wrapping real Laya (convaiinnovations/laya).
    Directly calls Router().predict(state, questions) with choice questions.
    """
    def __init__(self, default_model: str = "english"):
        self.default_model = default_model
        self.router = Router(default=default_model)
        self.model_name = "convaiinnovations/laya"
        self.criteria = DEPARTMENTS
        self.questions = {
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this customer support case?",
                "criteria": self.criteria
            }
        }

    def classify_case(
        self,
        subject: str,
        body: str,
        confidence_threshold: float = DEFAULT_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Runs real Laya inference with uncertainty gating.
        """
        state = f"Subject: {subject}\nDescription: {body}"
        
        t0 = time.perf_counter()
        res = self.router.predict(state, self.questions)
        lat_ms = (time.perf_counter() - t0) * 1000.0

        ans = res["answers"]["department"]
        predicted_dept = ans["choice"]
        confidence = round(float(ans.get("confidence", ans.get("answer_confidence", 0.0))), 4)
        probabilities = ans.get("probabilities", {})
        
        # Language/script routing from Laya Router
        routing_info = res.get("routing", {})
        detected_lang = routing_info.get("detection", {}).get("language", "English")
        if detected_lang == "en":
            detected_lang = "English"

        # Action policy gating
        if confidence >= confidence_threshold:
            action = "AUTO_ROUTE"
            routing_reason = f"Confidence {confidence:.2%} exceeds auto-route threshold {confidence_threshold:.2%}"
        else:
            action = "HUMAN_TRIAGE_ESCALATION"
            routing_reason = f"Confidence {confidence:.2%} below auto-route threshold {confidence_threshold:.2%} (flagged for review)"

        return {
            "predicted_department": predicted_dept,
            "confidence": confidence,
            "decision": action,
            "routing_reason": routing_reason,
            "language": detected_lang,
            "latency_ms": round(lat_ms, 2),
            "probabilities": probabilities,
            "model_identity": f"{self.model_name} ({res.get('routing', {}).get('model', self.default_model)})"
        }
