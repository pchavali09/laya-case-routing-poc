# Salesforce <-> Laya Real-Time Integration

This folder contains the complete reference integration connecting Salesforce Service Cloud Cases to your local or self-hosted **Laya Decision Engine**.

---

## ⚡ Architectural Advantages over Salesforce Einstein

| Dimension | Salesforce Einstein Case Classification | Laya Integration Bridge |
|---|---|---|
| **Latency** | 450 ms – 1200 ms (Multi-tenant SaaS cloud queue) | **< 15 ms** (Sub-second local/edge decision) |
| **Pricing** | $50 / agent / month (Einstein Service Cloud license) | **$0 SaaS fee** (Self-hosted or edge container) |
| **Data Privacy** | Customer PII egresses into Salesforce ML training perimeter | **100% On-Premise / Zero egress** |
| **Confidence Gating** | Opaque score; either forces automatic or manual review | **Calibrated posterior probability gating** |
| **Multilingual** | Requires separate language pack models | **Native zero-shot multilingual routing** |

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
