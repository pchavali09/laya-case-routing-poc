# Salesforce <-> Laya Real-Time Integration

This folder contains the complete reference integration connecting Salesforce Service Cloud Cases to your local or self-hosted **Laya Decision Engine**.

---

## ⚡ Architectural Comparison

| Dimension | Salesforce Einstein Case Classification | Local Laya Integration Bridge |
|---|---|---|
| **Deployment** | Multi-tenant SaaS cloud model | Self-hosted local/edge inference |
| **Model Artifact** | Closed Salesforce Einstein Sandbox model | `convaiinnovations/laya` (open-source weights) |
| **Inference Latency** | ~450 ms – 800 ms (Cloud network roundtrip) | ~58 ms (Apple M2 MPS local inference) |
| **Interface** | Salesforce standard Case layout & flow | FastAPI REST webhook + Apex Invocable Action |
| **173-Case Accuracy** | 99.42% (172/173 routed, 27 unrouted) | 48.55% (labels) to 63.01% (descriptive criteria) |

---

## 🚀 Quickstart: Running the Bridge

### 1. Launch the FastAPI Endpoint
```bash
make serve
# Or directly:
uvicorn salesforce.fastapi_bridge:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is immediately available at `http://localhost:8000/docs`.

### 2. Expose Locally for Salesforce Testing (ngrok or Cloudflare Tunnel)
```bash
ngrok http 8000
```

### 3. Configure in Salesforce
1. Navigate to **Setup** > **Security** > **Remote Site Settings** and add your tunnel URL.
2. In **Named Credentials**, create `Laya_Local_Engine` pointing to your URL.
3. Deploy `CaseLayaRoutingInvocable.cls` and invoke it from any **Salesforce Flow** or **Apex Trigger**.

### 4. Sample Curl Test
```bash
curl -X POST "http://localhost:8000/v1/cases/route" \
     -H "Content-Type: application/json" \
     -d '{
       "subject": "Overcharged on invoice #4812",
       "description": "We noticed duplicate $499 charges on our monthly statement.",
       "confidence_threshold": 0.85
     }'
```
Response:
```json
{
  "case_id": null,
  "predicted_department": "Billing",
  "confidence": 0.9658,
  "decision": "AUTO_ROUTE",
  "routing_reason": "Confidence 96.58% exceeds auto-route threshold 85.00%",
  "language": "English",
  "latency_ms": 11.4,
  "probabilities": {
    "Billing": 0.9658,
    "Technical": 0.0166,
    "Sales": 0.0129,
    "Operations": 0.003,
    "Other": 0.0017
  }
}
```
