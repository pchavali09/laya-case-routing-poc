"""
Salesforce <-> Laya Real-Time Integration Bridge
------------------------------------------------
FastAPI HTTP Webhook Service that receives incoming Salesforce Case payloads,
executes sub-15ms local Laya inference, and returns calibrated routing decisions.

Compatible with Salesforce External Services, Named Credentials, and Apex HTTP Callouts.
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import time

app = FastAPI(
    title="Laya Salesforce Case Routing Bridge",
    version="1.0.0",
    description="Local-first, sub-15ms decision engine bridge for Salesforce Case classification"
)

# Lazy-loaded inference engine
engine = None

def get_engine():
    global engine
    if engine is None:
        from src.laya_inference import LayaInferenceEngine
        engine = LayaInferenceEngine()
    return engine


class SalesforceCaseRequest(BaseModel):
    case_id: Optional[str] = Field(None, example="500XXXXXXXXXXXX")
    case_number: Optional[str] = Field(None, example="00010452")
    subject: str = Field(..., example="Unexpected charge on invoice #9401")
    description: str = Field(..., example="Our credit card was billed twice for the enterprise license renewal.")
    confidence_threshold: Optional[float] = Field(0.85, example=0.85)


class LayaRoutingResponse(BaseModel):
    case_id: Optional[str]
    predicted_department: str
    confidence: float
    decision: str
    routing_reason: str
    language: str
    latency_ms: float
    probabilities: Dict[str, float]


@app.get("/health")
def health_check():
    return {"status": "ok", "engine": "laya-local-first", "model": "convaiinnovations/laya:calibrated"}


@app.post("/v1/cases/route", response_model=LayaRoutingResponse)
def route_case(req: SalesforceCaseRequest):
    start_time = time.perf_counter()
    try:
        eng = get_engine()
        result = eng.classify_case(
            subject=req.subject,
            body=req.description,
            confidence_threshold=req.confidence_threshold or 0.85
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return LayaRoutingResponse(
            case_id=req.case_id,
            predicted_department=result["predicted_department"],
            confidence=result["confidence"],
            decision=result["decision"],
            routing_reason=result["routing_reason"],
            language=result["language"],
            latency_ms=round(elapsed_ms, 2),
            probabilities=result["probabilities"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
