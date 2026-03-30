"""
GST Compliance AI Engine — FastAPI Application
===============================================

Orchestrates all modules:
  1. HSN Semantic Retriever   (ML — embedding similarity)
  2. Compliance Rule Engine   (Deterministic — math + slab validation)
  3. Anomaly Detector         (ML — Isolation Forest)
  4. Autofill Predictor       (Statistical — frequency-based)
  5. Decision Engine          (Deterministic — weighted demerit scoring)
  6. Audit Logger             (Logging — JSON lines)

Models are loaded ONCE at startup via the lifespan context manager.
No per-request model loading.
"""

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI

from schemas import (
    InvoiceInput,
    InvoiceOutput,
    HSNResult,
    ComplianceResult,
    AnomalyResult,
    AutofillResult,
)
from models.hsn_classifier import HSNSemanticRetriever
from models.anomaly_detector import InvoiceAnomalyDetector
from models.autofill_predictor import AutofillPredictor
from rules.validation import validate_invoice
from rules.decision_engine import evaluate_decision
from services.audit_logger import log_request


# ---------- Startup / Shutdown ----------

_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy ML artifacts once. Release on shutdown."""
    print("[startup] Loading HSN Semantic Retriever …")
    _models["hsn"] = HSNSemanticRetriever()

    print("[startup] Loading Anomaly Detector …")
    _models["anomaly"] = InvoiceAnomalyDetector()

    print("[startup] Loading Autofill Predictor …")
    _models["autofill"] = AutofillPredictor()

    print("[startup] All models loaded. Server ready.")
    yield
    _models.clear()
    print("[shutdown] Models released.")


# ---------- App ----------

app = FastAPI(
    title="GST Compliance AI Engine",
    version="2.0.0",
    description="Hybrid rule-based + ML invoice validation backend.",
    lifespan=lifespan,
)


@app.post("/process-invoice", response_model=InvoiceOutput)
async def process_invoice(invoice: InvoiceInput):
    """
    Single endpoint that runs the full compliance pipeline:
      HSN retrieval → Rule validation → Anomaly detection → Autofill → Decision
    """

    # 1. HSN Semantic Retrieval (ML)
    hsn_res = _models["hsn"].predict(invoice.product_name)

    # 2. Compliance Rule Validation (Deterministic) — rate is percentage
    compliance_res = validate_invoice(invoice.value, invoice.rate, invoice.tax)

    # 3. Anomaly Detection (ML) — now uses customer_id as a feature
    anomaly_res = _models["anomaly"].detect(
        invoice.value, invoice.rate, invoice.tax, invoice.customer_id
    )

    # 4. Predictive Autofill (Statistical)
    autofill_res = _models["autofill"].predict(invoice.customer_id)

    # 5. Decision Engine (Deterministic scoring)
    decision, explanations = evaluate_decision(
        hsn_res, compliance_res, anomaly_res, autofill_res
    )

    # 6. Audit Log (every request logged for compliance trail)
    log_request(
        invoice_input=invoice.model_dump(),
        hsn=hsn_res,
        compliance=compliance_res,
        anomaly=anomaly_res,
        autofill=autofill_res,
        decision=decision,
        explanations=explanations,
    )

    return InvoiceOutput(
        hsn=HSNResult(**hsn_res),
        compliance=ComplianceResult(**compliance_res),
        anomaly=AnomalyResult(**anomaly_res),
        autofill=AutofillResult(**autofill_res),
        decision=decision,
        explanations=explanations,
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
