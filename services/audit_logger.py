"""
Audit Logger
============

Structured logging service for compliance audit trail.

Every API call logs:
1. Raw input payload
2. Per-module outputs (HSN, compliance, anomaly, autofill)
3. Final decision + explanations

Log format: JSON lines (one JSON object per line) for easy ingestion
into ELK/Splunk/BigQuery.

LIMITATION:
- Currently writes to a local file. In production, ship logs to a
  centralized logging service (e.g. Google Cloud Logging, AWS CloudWatch)
  with log rotation and retention policies.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure a dedicated file logger (separate from uvicorn's logger)
_logger = logging.getLogger("gst_audit")
_logger.setLevel(logging.INFO)
_logger.propagate = False

_handler = logging.FileHandler(LOG_DIR / "audit.jsonl", encoding="utf-8")
_handler.setFormatter(logging.Formatter("%(message)s"))
_logger.addHandler(_handler)


def log_request(invoice_input: dict, hsn: dict, compliance: dict,
                anomaly: dict, autofill: dict, decision: str,
                explanations: list[str]) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": invoice_input,
        "outputs": {
            "hsn": hsn,
            "compliance": compliance,
            "anomaly": anomaly,
            "autofill": autofill,
        },
        "decision": decision,
        "explanations": explanations,
    }
    _logger.info(json.dumps(record, default=str))
