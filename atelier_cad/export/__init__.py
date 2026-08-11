"""Export facade."""

from __future__ import annotations

import json
from typing import Any, Sequence

from atelier_cad.export.dxf import dxf_to_bytes, pieces_to_dxf
from atelier_cad.export.pdf import render_preview_figure, save_a0_pdf, save_tiled_pdf
from atelier_cad.export.schema import build_pattern_document
from atelier_cad.export.svg import pieces_to_svg
from atelier_cad.geometry import PatternPiece
from atelier_cad.measurements import FigureOptions, Measurements
from atelier_cad.pattern_spec import PatternSpec


def export_all(
    pieces: Sequence[PatternPiece],
    *,
    measurements: Measurements,
    figure: FigureOptions,
    spec: PatternSpec,
    title: str = "Atelier CAD",
) -> dict[str, Any]:
    laid = list(pieces)
    doc = build_pattern_document(
        pieces=laid,
        measurements=measurements,
        figure=figure,
        spec=spec,
        meta={"title": title},
    )
    return {
        "json": json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8"),
        "svg": pieces_to_svg(laid).encode("utf-8"),
        "dxf": dxf_to_bytes(pieces_to_dxf(laid)),
        "pdf": save_tiled_pdf(laid, title=title),
        "pdf_a0": save_a0_pdf(laid, title=title),
        "document": doc,
    }


__all__ = [
    "export_all",
    "build_pattern_document",
    "pieces_to_svg",
    "pieces_to_dxf",
    "dxf_to_bytes",
    "save_tiled_pdf",
    "save_a0_pdf",
    "render_preview_figure",
]
