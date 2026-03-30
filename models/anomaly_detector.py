"""
Invoice Anomaly Detection Engine
=================================

MODEL: Isolation Forest (sklearn)

WHY ISOLATION FOREST:
- Unsupervised — does not require labeled fraud data (which is rare and expensive).
- Performs well on high-dimensional numerical features with sparse anomalies.
- O(n log n) training, fast inference. Suitable for real-time API usage.
- Alternative considered: One-Class SVM — rejected because it scales poorly
  (O(n^2) to O(n^3)) and is sensitive to kernel choice.

INPUT FEATURES:
- value: invoice amount
- rate: GST rate (percentage)
- tax: tax charged
- customer_encoded: integer hash of customer_id (captures per-customer baseline)
- tx_frequency: number of transactions by this customer (pattern indicator)

WHY REPEATED DUMMY DATA IS BAD:
- Isolation Forest learns the "shape" of normal data. If training data is just
  the same 4 rows copied 10x, the model memorizes those exact points.
  Any input that deviates slightly from those 4 points gets flagged as anomaly,
  producing massive false-positive rates.
- In production, training data must be sampled from real historical invoices
  (ideally 10k+ diverse records) with natural variance in amounts, rates,
  and customer distribution.

LIMITATIONS:
- The mock training data below uses synthetic distributions that approximate
  real invoice patterns, but is NOT a substitute for real transaction data.
- customer_encoded uses a simple hash modulo — collisions exist.
  Production should use proper label encoding or frequency encoding.
- The model is static. In production, retrain periodically on a sliding window
  of recent transactions.
"""

import numpy as np
from sklearn.ensemble import IsolationForest


def _encode_customer(customer_id: str) -> int:
    """Deterministic integer encoding for customer_id.
    In production, replace with a proper label encoder backed by a lookup table."""
    return abs(hash(customer_id)) % 10000


# Mock transaction frequency lookup.
# In production: query from a time-windowed aggregate table.
_CUSTOMER_TX_COUNTS: dict[str, int] = {
    "CUST-001": 47,
    "CUST-002": 12,
    "CUST-003": 89,
    "CUST-004": 3,
}


class InvoiceAnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
        )

        # Diverse synthetic training data: [value, rate, tax, customer_encoded, tx_frequency]
        rng = np.random.RandomState(42)
        n_samples = 500

        values = rng.lognormal(mean=6.0, sigma=1.5, size=n_samples)  # range ~50 to ~50k
        rates = rng.choice([0, 5, 12, 18, 28], size=n_samples, p=[0.05, 0.2, 0.25, 0.35, 0.15]).astype(float)
        taxes = values * rates / 100
        # Add small gaussian noise to tax to simulate ERP rounding
        taxes += rng.normal(0, 0.3, size=n_samples)
        taxes = np.maximum(taxes, 0)

        cust_ids = rng.randint(0, 9999, size=n_samples).astype(float)
        tx_freqs = rng.poisson(lam=20, size=n_samples).astype(float)

        training_data = np.column_stack([values, rates, taxes, cust_ids, tx_freqs])
        self.model.fit(training_data)

    def detect(self, value: float, rate: float, tax: float, customer_id: str) -> dict:
        cust_encoded = float(_encode_customer(customer_id))
        tx_freq = float(_CUSTOMER_TX_COUNTS.get(customer_id, 0))

        feature_vector = np.array([[value, rate, tax, cust_encoded, tx_freq]])

        # decision_function returns the anomaly score: lower = more anomalous
        raw_score = float(self.model.decision_function(feature_vector)[0])
        prediction = int(self.model.predict(feature_vector)[0])  # 1 = normal, -1 = anomaly

        return {
            "anomaly_detected": prediction == -1,
            "anomaly_score": round(raw_score, 4),
        }
