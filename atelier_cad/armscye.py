"""EMKO-style armscye (пройма) control points and length — shared by bodice + sleeve."""

from __future__ import annotations

from atelier_cad.geometry import Point, cubic_bezier


def back_armscye_curve(
    shoulder: Point,
    *,
    w_back: float,
    w_arm: float,
    depth_g: float,
    n: int = 24,
) -> list[Point]:
    """
    Спинка: плечевая точка → пройма → низ проймы (центр рукава сетки).

    Контрольные точки приближают классическую сетку ЕМКО:
    - mid-back pitch чуть внутрь от линии ширины спинки
    - нижняя треть — более пологая к подрезу
    """
    underarm: Point = (w_back + w_arm / 2, depth_g)
    # Pitch: ~⅓ ширины проймы от линии спинки, ~55% глубины
    p1: Point = (w_back + w_arm * 0.22, depth_g * 0.42)
    p2: Point = (w_back + w_arm * 0.48, depth_g * 0.82)
    return cubic_bezier(shoulder, p1, p2, underarm, n=n)


def front_armscye_curve(
    shoulder: Point,
    underarm: Point,
    *,
    depth_g: float,
    n: int = 24,
) -> list[Point]:
    """
    Перед: плечевая → пройма → подрез.
    Передняя пройма глубже/круглее спинки (больше выемка).
    """
    ux, uy = underarm
    # Front scoop: control points pull curve toward CF-side of underarm
    p1: Point = (ux - 2.2, depth_g * 0.38)
    p2: Point = (ux - 0.6, depth_g * 0.78)
    # Keep shoulder Y in sync with curve start
    return cubic_bezier(shoulder, p1, p2, underarm, n=n)


def polyline_length(points: list[Point]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    return total


def estimate_armscye_half_lengths(
    *,
    w_back: float,
    w_arm: float,
    w_front: float,
    depth_g: float,
    sh_back: Point,
    sh_front: Point,
) -> tuple[float, float, float]:
    """
    Return (back_armscye_len, front_armscye_len, total_armscye_len) in cm
    for matching sleeve cap ease.
    """
    back = back_armscye_curve(sh_back, w_back=w_back, w_arm=w_arm, depth_g=depth_g)
    front_underarm = (w_front + w_arm / 2, depth_g)
    front = front_armscye_curve(sh_front, front_underarm, depth_g=depth_g)
    lb = polyline_length(back)
    lf = polyline_length(front)
    return lb, lf, lb + lf
