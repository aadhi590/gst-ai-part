"""
Predictive Autofill Engine
==========================

APPROACH: Statistical mode over recent customer transactions.

WHY NOT HEAVY ML:
- GST type and place of supply are highly deterministic per-customer.
  A customer shipping from Maharashtra almost always files IGST for
  inter-state and CGST+SGST for intra-state. The dominant pattern
  covers >90% of cases.
- A simple frequency-based lookup (mode of last N transactions) achieves
  near-perfect accuracy for repeat customers with zero training overhead.
- Heavy ML (e.g. sequence models) is justified only if customers frequently
  change supply patterns — which is rare in B2B GST.

FAILURE CASES:
- Unknown customer (no history) → returns null predictions with confidence 0.
- Customer with mixed patterns (e.g. ships to multiple states) → confidence
  reflects the dominance of the mode. A 55% confidence means the mode only
  appeared in 55% of transactions — the user should verify.

LIMITATIONS:
- Mock data below. In production, query from a transaction aggregate table
  with columns: customer_id, gst_type, place_of_supply, count.
"""


class AutofillPredictor:
    def __init__(self):
        # Mock aggregated history: {customer_id: {gst_type, place_of_supply, confidence}}
        # confidence = (mode_count / total_count) for that customer.
        self._customer_history = {
            "CUST-001": {
                "gst_type": "IGST",
                "place_of_supply": "27-Maharashtra",
                "confidence": 0.93,
            },
            "CUST-002": {
                "gst_type": "CGST_SGST",
                "place_of_supply": "29-Karnataka",
                "confidence": 0.88,
            },
            "CUST-003": {
                "gst_type": "IGST",
                "place_of_supply": "07-Delhi",
                "confidence": 0.71,
            },
            "CUST-004": {
                "gst_type": "CGST_SGST",
                "place_of_supply": "33-Tamil Nadu",
                "confidence": 0.60,
            },
        }

    def predict(self, customer_id: str) -> dict:
        if customer_id in self._customer_history:
            return self._customer_history[customer_id]

        # Unknown customer: explicitly return nulls rather than guessing.
        return {
            "gst_type": None,
            "place_of_supply": None,
            "confidence": 0.0,
        }
