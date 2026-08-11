"""Garment model registry: PatternSpec → PatternPiece list."""

from __future__ import annotations

from atelier_cad.geometry import PatternPiece, layout_pieces_horizontal
from atelier_cad.measurements import FigureOptions, Measurements
from atelier_cad.models.sheath_dress import draft_sheath_dress
from atelier_cad.models.skirt import draft_skirt
from atelier_cad.models.sleeve import draft_sleeve_set_in
from atelier_cad.pattern_spec import GarmentType, PatternSpec, PatternSpecError, parse_pattern_spec


SUPPORTED_V1 = {
    GarmentType.SKIRT_STRAIGHT,
    GarmentType.SKIRT_A_LINE,
    GarmentType.DRESS_SHEATH,
    GarmentType.SLEEVE_SET_IN,
}


def build_from_spec(
    measurements: Measurements,
    spec: PatternSpec | dict,
    figure: FigureOptions | None = None,
    *,
    layout: bool = True,
) -> list[PatternPiece]:
    """
    Build parametric pattern pieces from a validated PatternSpec.
    Raises PatternSpecError on unsupported / invalid specs (no broken PDF).
    """
    parsed = parse_pattern_spec(spec)
    fig = figure or FigureOptions()

    errors = measurements.validate()
    if errors:
        raise PatternSpecError("; ".join(errors))

    if parsed.garment_type not in SUPPORTED_V1:
        raise PatternSpecError(
            f"Модель «{parsed.garment_type.value}» ещё не поддерживается в v1. "
            f"Доступно: {[g.value for g in sorted(SUPPORTED_V1, key=lambda g: g.value)]}"
        )

    if parsed.garment_type in (GarmentType.SKIRT_STRAIGHT, GarmentType.SKIRT_A_LINE):
        pieces = draft_skirt(measurements, parsed, fig)
    elif parsed.garment_type == GarmentType.DRESS_SHEATH:
        pieces = draft_sheath_dress(measurements, parsed, fig)
    elif parsed.garment_type == GarmentType.SLEEVE_SET_IN:
        pieces = draft_sleeve_set_in(measurements, parsed, fig)
    else:
        raise PatternSpecError(f"Нет шаблона для {parsed.garment_type}")

    if not pieces:
        raise PatternSpecError("Построение вернуло пустой набор деталей")

    if layout:
        return layout_pieces_horizontal(pieces)
    return pieces
