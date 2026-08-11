#!/usr/bin/env python3
"""Smoke: py_compile + build one skirt PDF without Streamlit / OpenRouter key."""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def main() -> int:
    modules = [
        ROOT / "app.py",
        ROOT / "atelier_cad" / "measurements.py",
        ROOT / "atelier_cad" / "drafting.py",
        ROOT / "atelier_cad" / "geometry.py",
        ROOT / "atelier_cad" / "pattern_spec.py",
        ROOT / "atelier_cad" / "openrouter.py",
        ROOT / "atelier_cad" / "armscye.py",
        ROOT / "atelier_cad" / "models" / "skirt.py",
        ROOT / "atelier_cad" / "models" / "sheath_dress.py",
        ROOT / "atelier_cad" / "models" / "sleeve.py",
        ROOT / "atelier_cad" / "export" / "pdf.py",
        ROOT / "atelier_cad" / "export" / "svg.py",
        ROOT / "atelier_cad" / "export" / "dxf.py",
    ]
    for path in modules:
        py_compile.compile(str(path), doraise=True)
        print(f"py_compile OK: {path.relative_to(ROOT)}")

    from atelier_cad.export import export_all
    from atelier_cad.measurements import DEFAULT_MEASUREMENTS, FigureOptions
    from atelier_cad.models import build_from_spec
    from atelier_cad.pattern_spec import (
        preset_dress_sheath_boat_midi,
        preset_dress_sheath_with_sleeve,
        preset_skirt_straight,
        preset_sleeve_set_in_long,
    )

    m = DEFAULT_MEASUREMENTS
    fig = FigureOptions()

    for label, spec in (
        ("skirt_straight", preset_skirt_straight()),
        ("dress_sheath_boat_midi", preset_dress_sheath_boat_midi()),
        ("sleeve_set_in_long", preset_sleeve_set_in_long()),
        ("dress_sheath_with_sleeve", preset_dress_sheath_with_sleeve()),
    ):
        pieces = build_from_spec(m, spec, fig)
        bundle = export_all(
            pieces,
            measurements=m,
            figure=fig,
            spec=spec,
            title=f"Atelier CAD smoke — {label}",
        )
        pdf_path = OUTPUTS / f"smoke_{label}.pdf"
        svg_path = OUTPUTS / f"smoke_{label}.svg"
        json_path = OUTPUTS / f"smoke_{label}.json"
        dxf_path = OUTPUTS / f"smoke_{label}.dxf"
        pdf_path.write_bytes(bundle["pdf"])
        svg_path.write_bytes(bundle["svg"])
        json_path.write_bytes(bundle["json"])
        dxf_path.write_bytes(bundle["dxf"])
        a0_path = OUTPUTS / f"smoke_{label}_A0.pdf"
        a0_path.write_bytes(bundle["pdf_a0"])
        print(
            f"built {label}: pieces={len(pieces)} "
            f"pdf={pdf_path.stat().st_size}B a0={a0_path.stat().st_size}B "
            f"svg={svg_path.stat().st_size}B"
        )

    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
