"""DXF export with CUT / SEW / GRAIN / TEXT layers."""

from __future__ import annotations

import io

import ezdxf
from ezdxf import units

from atelier_cad.geometry import PatternPiece


LAYER_DEFS = (
    ("CUT", 1),  # red
    ("SEW", 5),  # blue
    ("GRAIN", 8),  # gray
    ("TEXT", 7),  # white/black
    ("NOTCH", 3),  # green
)


def _ensure_layers(doc) -> None:
    for name, color in LAYER_DEFS:
        if name not in doc.layers:
            doc.layers.add(name, color=color)


def pieces_to_dxf(pieces: list[PatternPiece]):
    doc = ezdxf.new("R2010")
    doc.units = units.CM
    _ensure_layers(doc)
    msp = doc.modelspace()

    for piece in pieces:
        if len(piece.cut_outline) >= 2:
            msp.add_lwpolyline(
                piece.cut_outline,
                close=True,
                dxfattribs={"layer": "CUT"},
            )
        if piece.sew_outline and len(piece.sew_outline) >= 2:
            msp.add_lwpolyline(
                piece.sew_outline,
                close=True,
                dxfattribs={"layer": "SEW"},
            )
        for dart in piece.darts:
            if len(dart) >= 2:
                msp.add_lwpolyline(dart, close=False, dxfattribs={"layer": "SEW"})
        if piece.grainline:
            msp.add_line(piece.grainline[0], piece.grainline[1], dxfattribs={"layer": "GRAIN"})
        for nx, ny in piece.notches:
            msp.add_line((nx - 0.4, ny), (nx + 0.4, ny), dxfattribs={"layer": "NOTCH"})
        for (x, y), text in piece.labels:
            msp.add_text(text, dxfattribs={"layer": "TEXT", "height": 0.8}).set_placement((x, y))
        msp.add_text(piece.name, dxfattribs={"layer": "TEXT", "height": 1.0}).set_placement(
            piece.cut_outline[0]
        )

    return doc


def dxf_to_bytes(doc) -> bytes:
    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode("utf-8")
