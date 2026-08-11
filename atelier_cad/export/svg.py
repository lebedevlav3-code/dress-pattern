"""SVG export for pattern pieces."""

from __future__ import annotations

from atelier_cad.geometry import PatternPiece, union_bounds


def _path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    parts = [f"M {points[0][0]:.3f} {points[0][1]:.3f}"]
    for x, y in points[1:]:
        parts.append(f"L {x:.3f} {y:.3f}")
    return " ".join(parts)


def pieces_to_svg(pieces: list[PatternPiece], *, scale: float = 10.0) -> str:
    """
    Export pieces to SVG. Coordinates in cm; viewBox scaled by `scale` (px per cm).
    Y axis is kept as in CAD (down positive) — SVG will look upright for sewing.
    """
    width_cm, height_cm = union_bounds(pieces, padding=2.0)
    w = width_cm * scale
    h = height_cm * scale

    layers: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.1f}" height="{h:.1f}" '
        f'viewBox="0 0 {width_cm:.2f} {height_cm:.2f}">',
        '<g id="CUT" fill="none" stroke="#111" stroke-width="0.05">',
    ]
    for piece in pieces:
        layers.append(f'  <path d="{_path(piece.cut_outline)} Z" data-piece="{piece.name}"/>')
    layers.append("</g>")

    layers.append('<g id="SEW" fill="none" stroke="#3366cc" stroke-width="0.04" stroke-dasharray="0.3 0.2">')
    for piece in pieces:
        if piece.sew_outline:
            layers.append(f'  <path d="{_path(piece.sew_outline)} Z"/>')
        for dart in piece.darts:
            layers.append(f'  <path d="{_path(dart)}"/>')
    layers.append("</g>")

    layers.append('<g id="GRAIN" stroke="#888" stroke-width="0.03">')
    for piece in pieces:
        if piece.grainline:
            (x1, y1), (x2, y2) = piece.grainline
            layers.append(f'  <line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}"/>')
            layers.append(
                f'  <text x="{(x1+x2)/2:.3f}" y="{(y1+y2)/2 - 0.4:.3f}" '
                f'font-size="0.6" fill="#666">ДОЛЕВАЯ</text>'
            )
    layers.append("</g>")

    layers.append('<g id="TEXT" font-size="0.7" fill="#222">')
    for piece in pieces:
        for (x, y), text in piece.labels:
            layers.append(f'  <text x="{x:.3f}" y="{y:.3f}">{_escape(text)}</text>')
        for nx, ny in piece.notches:
            layers.append(
                f'  <line x1="{nx-0.4:.3f}" y1="{ny:.3f}" x2="{nx+0.4:.3f}" y2="{ny:.3f}" '
                f'stroke="#c00" stroke-width="0.05"/>'
            )
    layers.append("</g>")
    layers.append("</svg>")
    return "\n".join(layers)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
