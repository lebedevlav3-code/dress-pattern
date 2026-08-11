"""EMKO / Müller simplified grid calculation (bodice foundation)."""

from __future__ import annotations

from typing import Any, TypedDict

from atelier_cad.measurements import FigureOptions, Measurements


class Widths(TypedDict):
    total: float
    back: float
    arm: float
    front: float


class Levels(TypedDict):
    A: float
    G: float
    T: float
    B: float
    N: float


class Darts(TypedDict):
    back: float
    side: float
    front: float


class Grid(TypedDict):
    W: Widths
    L: Levels
    D: Darts
    Misc: dict[str, float]


def calculate_grid(m: Measurements, opts: FigureOptions | dict[str, str]) -> Grid:
    """Compute bodice construction grid from measurements and figure options."""
    if isinstance(opts, FigureOptions):
        o = opts.to_dict()
    else:
        o = opts

    total_w = (m.OG + m.Pruh) / 2
    w_back = (m.OG / 8) + 5.5
    if o.get("bust") == "полная":
        w_back -= 0.5

    w_arm = (m.OG / 8) - 1.5
    if w_arm < 9.5:
        w_arm = 9.5

    w_front = total_w - w_back - w_arm

    depth_arm = (m.OG / 10) + 10.5 + 2.5
    if o.get("shoulder") == "покатые":
        depth_arm += 1.5
    elif o.get("shoulder") == "прямые":
        depth_arm -= 1.0

    # Hip level: ~18–20 cm below waist (classic EMKO approximation)
    hip_drop = 19.0
    levels: Levels = {
        "A": 0.0,
        "G": depth_arm,
        "T": m.DTS,
        "B": m.DTS + hip_drop,
        "N": m.DI,
    }

    w_waist_needed = (m.OT / 2) + (m.Ptal / 2)
    total_dart = max(0.0, total_w - w_waist_needed)
    darts: Darts = {
        "back": total_dart * 0.25,
        "side": total_dart * 0.45,
        "front": total_dart * 0.30,
    }

    bust_dart = 2.0
    if m.OG > 90:
        bust_dart = 3.5
    if m.OG > 105:
        bust_dart = 5.0
    if o.get("bust") == "полная":
        bust_dart += 1.5

    return {
        "W": {
            "total": total_w,
            "back": w_back,
            "arm": w_arm,
            "front": w_front,
        },
        "L": levels,
        "D": darts,
        "Misc": {"bust_dart": bust_dart},
    }


def grid_to_public_dict(grid: Grid) -> dict[str, Any]:
    """JSON-serialisable copy of the grid."""
    return {
        "W": dict(grid["W"]),
        "L": dict(grid["L"]),
        "D": dict(grid["D"]),
        "Misc": dict(grid["Misc"]),
    }
