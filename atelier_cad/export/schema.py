"""Internal JSON pattern schema for interchange."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from atelier_cad.geometry import PatternPiece
from atelier_cad.measurements import FigureOptions, Measurements
from atelier_cad.pattern_spec import PatternSpec


SCHEMA_VERSION = "1.0.0"


def piece_to_dict(piece: PatternPiece) -> dict[str, Any]:
    return {
        "name": piece.name,
        "cut_outline_cm": piece.cut_outline,
        "sew_outline_cm": piece.sew_outline,
        "darts_cm": piece.darts,
        "notches_cm": piece.notches,
        "grainline_cm": list(piece.grainline) if piece.grainline else None,
        "labels": [{"at_cm": list(at), "text": text} for at, text in piece.labels],
        "seam_allowance_cm": piece.seam_allowance_cm,
        "hem_allowance_cm": piece.hem_allowance_cm,
    }


def build_pattern_document(
    *,
    pieces: list[PatternPiece],
    measurements: Measurements,
    figure: FigureOptions,
    spec: PatternSpec,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "units": "cm",
        "measurements": measurements.to_dict(),
        "figure_options": figure.to_dict(),
        "pattern_spec": spec.to_public_dict(),
        "pieces": [piece_to_dict(p) for p in pieces],
        "meta": meta or {},
    }
