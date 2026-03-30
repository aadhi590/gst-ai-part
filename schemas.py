from pydantic import BaseModel, Field
from typing import Optional, List


class InvoiceInput(BaseModel):
    """
    rate: GST rate as a PERCENTAGE (e.g. 18 means 18%).
    tax:  The tax amount charged on the invoice.
    Formula enforced: tax == value * rate / 100
    """
    product_name: str
    value: float
    rate: float = Field(..., description="GST rate as percentage, e.g. 18 for 18%")
    tax: float
    customer_id: str


class HSNResult(BaseModel):
    code: Optional[str]
    confidence: float
    matched_description: str
    fallback_required: bool


class ComplianceResult(BaseModel):
    hard_errors: List[str]
    soft_alerts: List[str]
    is_valid: bool


class AnomalyResult(BaseModel):
    anomaly_detected: bool
    anomaly_score: float


class AutofillResult(BaseModel):
    gst_type: Optional[str]
    place_of_supply: Optional[str]
    confidence: float


class InvoiceOutput(BaseModel):
    hsn: HSNResult
    compliance: ComplianceResult
    anomaly: AnomalyResult
    autofill: AutofillResult
    decision: str  # "approved" | "review_required" | "blocked"
    explanations: List[str]
