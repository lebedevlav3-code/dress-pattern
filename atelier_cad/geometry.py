"""Pattern piece geometry: contours, darts, notches, grain, seam allowance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union


Point = tuple[float, float]


@dataclass
class PatternPiece:
    """Single pattern piece in centimetres (construction origin top-left, Y down)."""

    name: str
    cut_outline: list[Point]
    sew_outline: list[Point] | None = None
    darts: list[list[Point]] = field(default_factory=list)
    notches: list[Point] = field(default_factory=list)
    grainline: tuple[Point, Point] | None = None
    labels: list[tuple[Point, str]] = field(default_factory=list)
    seam_allowance_cm: float = 1.5
    hem_allowance_cm: float = 4.0

    def bounds(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.cut_outline]
        ys = [p[1] for p in self.cut_outline]
        return min(xs), min(ys), max(xs), max(ys)


def quadratic_bezier(p0: Point, p1: Point, p2: Point, n: int = 16) -> list[Point]:
    """Sample a quadratic Bezier curve (used for neckline / armscye splines)."""
    t = np.linspace(0.0, 1.0, n)
    pts: list[Point] = []
    for ti in t:
        x = (1 - ti) ** 2 * p0[0] + 2 * (1 - ti) * ti * p1[0] + ti**2 * p2[0]
        y = (1 - ti) ** 2 * p0[1] + 2 * (1 - ti) * ti * p1[1] + ti**2 * p2[1]
        pts.append((float(x), float(y)))
    return pts


def cubic_bezier(p0: Point, p1: Point, p2: Point, p3: Point, n: int = 20) -> list[Point]:
    t = np.linspace(0.0, 1.0, n)
    pts: list[Point] = []
    for ti in t:
        mt = 1 - ti
        x = (
            mt**3 * p0[0]
            + 3 * mt**2 * ti * p1[0]
            + 3 * mt * ti**2 * p2[0]
            + ti**3 * p3[0]
        )
        y = (
            mt**3 * p0[1]
            + 3 * mt**2 * ti * p1[1]
            + 3 * mt * ti**2 * p2[1]
            + ti**3 * p3[1]
        )
        pts.append((float(x), float(y)))
    return pts


def close_ring(points: Sequence[Point]) -> list[Point]:
    pts = list(points)
    if pts and pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def apply_seam_allowance(
    outline: Sequence[Point],
    allowance_cm: float,
    *,
    join_style: int = 2,
) -> list[Point]:
    """Offset closed outline outward using shapely.buffer (cm)."""
    if allowance_cm <= 0:
        return list(outline)
    ring = close_ring(outline)
    poly = Polygon(ring)
    if not poly.is_valid or poly.area <= 0:
        # Fall back to LineString buffer for open / invalid rings
        line = LineString(outline)
        buffered = line.buffer(allowance_cm, join_style=join_style, cap_style=2)
        if buffered.is_empty:
            return list(outline)
        if buffered.geom_type == "Polygon":
            return [(float(x), float(y)) for x, y in buffered.exterior.coords]
        return list(outline)

    buffered = poly.buffer(allowance_cm, join_style=join_style, cap_style=1)
    if buffered.is_empty:
        return list(outline)
    geom = buffered
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    return [(float(x), float(y)) for x, y in geom.exterior.coords]


def notch_marks(points: Iterable[Point], size_cm: float = 0.5) -> list[tuple[Point, Point]]:
    """Return short tick segments for notch visualisation (horizontal)."""
    marks: list[tuple[Point, Point]] = []
    for x, y in points:
        marks.append(((x - size_cm, y), (x + size_cm, y)))
    return marks


def union_bounds(pieces: Sequence[PatternPiece], padding: float = 2.0) -> tuple[float, float]:
    """Return (width_cm, height_cm) covering all pieces with padding."""
    if not pieces:
        return 10.0, 10.0
    min_x = min(p.bounds()[0] for p in pieces)
    min_y = min(p.bounds()[1] for p in pieces)
    max_x = max(p.bounds()[2] for p in pieces)
    max_y = max(p.bounds()[3] for p in pieces)
    return (max_x - min_x) + 2 * padding, (max_y - min_y) + 2 * padding


def translate_piece(piece: PatternPiece, dx: float, dy: float) -> PatternPiece:
    def t(pts: list[Point]) -> list[Point]:
        return [(x + dx, y + dy) for x, y in pts]

    grain = None
    if piece.grainline:
        (x1, y1), (x2, y2) = piece.grainline
        grain = ((x1 + dx, y1 + dy), (x2 + dx, y2 + dy))

    return PatternPiece(
        name=piece.name,
        cut_outline=t(piece.cut_outline),
        sew_outline=t(piece.sew_outline) if piece.sew_outline else None,
        darts=[t(d) for d in piece.darts],
        notches=t(piece.notches),
        grainline=grain,
        labels=[((x + dx, y + dy), text) for (x, y), text in piece.labels],
        seam_allowance_cm=piece.seam_allowance_cm,
        hem_allowance_cm=piece.hem_allowance_cm,
    )


def layout_pieces_horizontal(
    pieces: Sequence[PatternPiece], gap_cm: float = 3.0
) -> list[PatternPiece]:
    """Place pieces side by side without overlap for print."""
    laid: list[PatternPiece] = []
    cursor_x = 0.0
    for piece in pieces:
        min_x, min_y, max_x, max_y = piece.bounds()
        dx = cursor_x - min_x
        dy = -min_y
        moved = translate_piece(piece, dx, dy)
        laid.append(moved)
        cursor_x += (max_x - min_x) + gap_cm
    return laid


def merge_piece_polygons(pieces: Sequence[PatternPiece]) -> Polygon | None:
    polys = []
    for p in pieces:
        ring = close_ring(p.cut_outline)
        poly = Polygon(ring)
        if poly.is_valid and poly.area > 0:
            polys.append(poly)
    if not polys:
        return None
    return unary_union(polys)
