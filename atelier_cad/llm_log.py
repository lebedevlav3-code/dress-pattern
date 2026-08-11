"""LLM request logging without PII (no measurements, names, photos)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger("atelier_cad.llm")

# Default log file (gitignored via outputs/)
DEFAULT_LOG_PATH = Path("outputs/llm.log")


def _ensure_logger(path: Path = DEFAULT_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not any(isinstance(h, logging.FileHandler) for h in LOG.handlers):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        LOG.addHandler(handler)
        LOG.setLevel(logging.INFO)


def log_llm_event(
    *,
    event: str,
    model: str | None = None,
    prompt_chars: int | None = None,
    response_chars: int | None = None,
    garment_type: str | None = None,
    ok: bool | None = None,
    error_class: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Append one JSON line. Never include measurements, names, emails, or image bytes.
    """
    _ensure_logger()
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
    }
    if model:
        payload["model"] = model
    if prompt_chars is not None:
        payload["prompt_chars"] = prompt_chars
    if response_chars is not None:
        payload["response_chars"] = response_chars
    if garment_type:
        payload["garment_type"] = garment_type
    if ok is not None:
        payload["ok"] = ok
    if error_class:
        payload["error_class"] = error_class
    if extra:
        # Strip obvious PII keys if caller slipped
        safe = {
            k: v
            for k, v in extra.items()
            if k.lower() not in {"measurements", "name", "email", "phone", "photo", "image", "pii"}
        }
        payload["extra"] = safe
    LOG.info(json.dumps(payload, ensure_ascii=False))
