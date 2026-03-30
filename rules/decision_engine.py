"""
Decision Engine
===============

Combines outputs from all modules into a single auditable verdict.

DECISION LOGIC (not naive if-else):
- Uses a weighted demerit scoring system.
- Each risk signal contributes a demerit score proportional to its severity.
- Thresholds:
    total_demerits >= BLOCK_THRESHOLD  → "blocked"
    total_demerits >= REVIEW_THRESHOLD → "review_required"
    else                               → "approved"

WHY SCORING OVER IF-ELSE:
- Simple if-else chains create brittle priority ladders. Adding a new signal
  requires restructuring the entire chain.
- Weighted scoring allows composable risk: two medium-risk signals can
  together trigger review, even if neither alone would.
- Weights can be tuned from audit feedback without code changes.

OVERRIDE:
- Compliance hard errors bypass scoring entirely and force "blocked".
  Deterministic rules are non-negotiable.

LIMITATION:
- Weights below are manually set heuristics. In production, calibrate
  weights using historical audit outcomes (logistic regression on
  demerit features → actual fraud/error labels).
"""

# Demerit weights (tunable)
WEIGHT_COMPLIANCE_SOFT = 10        # per soft alert
WEIGHT_HSN_LOW_CONFIDENCE = 25
WEIGHT_HSN_FALLBACK = 40           # fallback means confidence is very low
WEIGHT_ANOMALY_DETECTED = 30
WEIGHT_UNKNOWN_CUSTOMER = 15

REVIEW_THRESHOLD = 20
BLOCK_THRESHOLD = 60


def evaluate_decision(
    hsn_result: dict,
    compliance_result: dict,
    anomaly_result: dict,
    autofill_result: dict,
) -> tuple[str, list[str]]:
    explanations: list[str] = []

    # ── OVERRIDE: Deterministic compliance failures are non-negotiable ──
    if not compliance_result["is_valid"]:
        explanations.append("BLOCKED: Compliance rule violations detected.")
        for err in compliance_result["hard_errors"]:
            explanations.append(f"  → {err}")
        return "blocked", explanations

    # ── Weighted demerit scoring ──
    total_demerits = 0

    # Soft compliance alerts
    n_soft = len(compliance_result["soft_alerts"])
    if n_soft > 0:
        penalty = n_soft * WEIGHT_COMPLIANCE_SOFT
        total_demerits += penalty
        for alert in compliance_result["soft_alerts"]:
            explanations.append(f"SOFT_ALERT (+{WEIGHT_COMPLIANCE_SOFT}): {alert}")

    # HSN confidence
    if hsn_result["fallback_required"]:
        total_demerits += WEIGHT_HSN_FALLBACK
        explanations.append(
            f"HSN_FALLBACK (+{WEIGHT_HSN_FALLBACK}): Confidence "
            f"{hsn_result['confidence']} is below threshold. "
            f"Manual HSN classification required."
        )
    elif hsn_result["confidence"] < 0.70:
        total_demerits += WEIGHT_HSN_LOW_CONFIDENCE
        explanations.append(
            f"HSN_LOW_CONF (+{WEIGHT_HSN_LOW_CONFIDENCE}): HSN match "
            f"confidence {hsn_result['confidence']} is moderate. "
            f"Matched: '{hsn_result['matched_description']}'."
        )

    # Anomaly
    if anomaly_result["anomaly_detected"]:
        total_demerits += WEIGHT_ANOMALY_DETECTED
        explanations.append(
            f"ANOMALY (+{WEIGHT_ANOMALY_DETECTED}): ML anomaly detector "
            f"flagged this transaction (score: {anomaly_result['anomaly_score']})."
        )

    # Unknown customer
    if autofill_result["confidence"] == 0.0:
        total_demerits += WEIGHT_UNKNOWN_CUSTOMER
        explanations.append(
            f"UNKNOWN_CUSTOMER (+{WEIGHT_UNKNOWN_CUSTOMER}): No transaction "
            f"history found. Autofill disabled."
        )

    # ── Final verdict ──
    if total_demerits >= BLOCK_THRESHOLD:
        explanations.insert(0, f"BLOCKED: Total demerit score {total_demerits} >= {BLOCK_THRESHOLD}.")
        return "blocked", explanations
    elif total_demerits >= REVIEW_THRESHOLD:
        explanations.insert(0, f"REVIEW_REQUIRED: Total demerit score {total_demerits} >= {REVIEW_THRESHOLD}.")
        return "review_required", explanations
    else:
        explanations.insert(0, f"APPROVED: Total demerit score {total_demerits} is within acceptable limits.")
        return "approved", explanations
