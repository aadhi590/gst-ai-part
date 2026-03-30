"""
Compliance Rule Engine
======================

This module contains ONLY deterministic, mathematically provable rules.
No ML. No probabilities. No confidence scores.

RATE FORMAT CONVENTION (CRITICAL):
- `rate` is a PERCENTAGE: 18 means 18%, 5 means 5%.
- Formula: expected_tax = value * rate / 100
- This convention is enforced across the entire system.
  If a caller passes 0.18 meaning 18%, it will be caught as an invalid slab.

VALID GST SLABS (as of 2024):
- 0% (exempt)
- 5%
- 12%
- 18%
- 28%
- Cess rates exist but are item-specific and handled separately in production.

DESIGN:
- This module's output OVERRIDES all ML decisions.
  If `is_valid` is False, the decision engine must block the invoice
  regardless of what the anomaly detector or HSN retriever says.
"""

VALID_GST_SLABS = {0, 5, 12, 18, 28}

# Allow rounding tolerance of 1 rupee to accommodate ERP float arithmetic
TAX_EPSILON = 1.0


def validate_invoice(value: float, rate: float, tax: float) -> dict:
    hard_errors: list[str] = []
    soft_alerts: list[str] = []

    # --- Hard errors: must be fixed before submission ---

    # 1. Negative or zero value
    if value <= 0:
        hard_errors.append(f"Invoice value must be positive. Got: {value}")

    # 2. Negative rate
    if rate < 0:
        hard_errors.append(f"GST rate cannot be negative. Got: {rate}")

    # 3. Negative tax
    if tax < 0:
        hard_errors.append(f"Tax amount cannot be negative. Got: {tax}")

    # 4. Tax calculation mismatch  (value * rate / 100)
    if value > 0 and rate >= 0:
        expected_tax = value * rate / 100
        diff = abs(expected_tax - tax)
        if diff > TAX_EPSILON:
            hard_errors.append(
                f"Tax mismatch: expected {round(expected_tax, 2)} "
                f"(= {value} × {rate}/100), got {tax}. "
                f"Difference: {round(diff, 2)}"
            )

    # --- Soft alerts: suspicious but not blocking ---

    # 5. Non-standard GST slab
    if rate not in VALID_GST_SLABS:
        soft_alerts.append(
            f"Rate {rate}% is not a standard GST slab {sorted(VALID_GST_SLABS)}. "
            f"Verify if cess or special rate applies."
        )

    # 6. Unusually high invoice value
    if value > 1_000_000:
        soft_alerts.append(
            f"Invoice value ₹{value:,.2f} exceeds ₹10,00,000. "
            f"Flag for manual review."
        )

    # 7. Zero-rated but non-zero tax
    if rate == 0 and tax > 0:
        hard_errors.append(
            f"Rate is 0% but tax charged is {tax}. "
            f"Zero-rated supplies must have zero tax."
        )

    return {
        "hard_errors": hard_errors,
        "soft_alerts": soft_alerts,
        "is_valid": len(hard_errors) == 0,
    }
