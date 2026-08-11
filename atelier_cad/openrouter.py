"""OpenRouter client (OpenAI-compatible) for PatternSpec structured output."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from atelier_cad.llm_log import log_llm_event
from atelier_cad.pattern_spec import PatternSpec, PatternSpecError, parse_pattern_spec

DEFAULT_MODEL = "openai/gpt-4o-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class OpenRouterNotConfigured(RuntimeError):
    """Raised when OPENROUTER_API_KEY is missing."""


def _load_system_prompt() -> str:
    path = PROMPTS_DIR / "pattern_spec_system.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return (
        "Ты помогаешь заполнить PatternSpec (JSON параметров швейной модели). "
        "Не рисуй SVG/выкройку. Только параметры."
    )


def get_api_key() -> str | None:
    return os.getenv("OPENROUTER_API_KEY") or None


def is_configured() -> bool:
    return bool(get_api_key())


def pattern_spec_json_schema() -> dict[str, Any]:
    """JSON Schema for OpenRouter structured outputs."""
    return PatternSpec.model_json_schema()


def text_to_pattern_spec(
    description: str,
    *,
    model: str | None = None,
    temperature: float = 0.2,
) -> PatternSpec:
    """
    Call OpenRouter to fill PatternSpec from a Russian garment description.
    Does NOT send body measurements (PII). Measurements are applied later in CAD.
    """
    api_key = get_api_key()
    if not api_key:
        raise OpenRouterNotConfigured(
            "OPENROUTER_API_KEY не задан. Используйте ручной PatternSpec или пресет."
        )

    model_id = model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    system = _load_system_prompt()
    user = (
        "Описание изделия от ученицы (без мерок):\n"
        f"{description.strip()}\n\n"
        "Верни только объект PatternSpec по схеме."
    )

    log_llm_event(
        event="pattern_spec_request",
        model=model_id,
        prompt_chars=len(system) + len(user),
    )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Пакет openai не установлен") from exc

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    schema = pattern_spec_json_schema()

    try:
        completion = client.chat.completions.create(
            model=model_id,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "pattern_spec",
                    "strict": True,
                    "schema": _make_schema_strict(schema),
                },
            },
            extra_headers={
                "HTTP-Referer": "https://github.com/lebedevlav3-code/dress-pattern",
                "X-Title": "Atelier CAD",
            },
        )
        content = completion.choices[0].message.content or ""
        log_llm_event(
            event="pattern_spec_response",
            model=model_id,
            response_chars=len(content),
            ok=True,
        )
        data = json.loads(content)
        data["source"] = "openrouter"
        return parse_pattern_spec(data)
    except PatternSpecError:
        log_llm_event(
            event="pattern_spec_invalid",
            model=model_id,
            ok=False,
            error_class="PatternSpecError",
        )
        raise
    except Exception as exc:  # noqa: BLE001 — surface as PatternSpecError to UI
        log_llm_event(
            event="pattern_spec_error",
            model=model_id,
            ok=False,
            error_class=type(exc).__name__,
        )
        raise PatternSpecError(f"OpenRouter не смог построить PatternSpec: {exc}") from exc


def _make_schema_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """
    OpenAI-style strict JSON schema needs additionalProperties:false on objects.
    Pydantic schemas are close; we shallow-fix common gaps.
    """
    out = json.loads(json.dumps(schema))

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node.setdefault("additionalProperties", False)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(out)
    return out


def describe_to_spec_or_preset(description: str) -> PatternSpec:
    """
    Prefer OpenRouter; if key missing, heuristic map for smoke / offline demo.
    """
    if is_configured():
        return text_to_pattern_spec(description)

    from atelier_cad.pattern_spec import (
        preset_dress_sheath_boat_midi,
        preset_dress_sheath_with_sleeve,
        preset_skirt_a_line,
        preset_skirt_straight,
        preset_sleeve_set_in_long,
    )

    low = description.lower()
    log_llm_event(event="pattern_spec_offline_heuristic", ok=True, extra={"chars": len(description)})
    if "рукав" in low and "футляр" not in low and "плать" not in low:
        return preset_sleeve_set_in_long()
    if ("футляр" in low or "плать" in low) and "рукав" in low:
        return preset_dress_sheath_with_sleeve()
    if "футляр" in low or "плать" in low:
        return preset_dress_sheath_boat_midi()
    if "трапец" in low or "a-line" in low or "а-силуэт" in low:
        return preset_skirt_a_line()
    return preset_skirt_straight()
