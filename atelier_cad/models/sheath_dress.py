"""Sheath dress (платье-футляр) — bodice+skirt block from EMKO grid."""

from __future__ import annotations

import numpy as np

from atelier_cad.armscye import back_armscye_curve, front_armscye_curve
from atelier_cad.drafting import calculate_grid
from atelier_cad.geometry import PatternPiece, apply_seam_allowance, quadratic_bezier
from atelier_cad.measurements import FigureOptions, Measurements
from atelier_cad.pattern_spec import Neckline, PatternSpec, SleeveStyle, resolve_length_cm


def draft_sheath_dress(
    m: Measurements,
    spec: PatternSpec,
    figure: FigureOptions | None = None,
) -> list[PatternPiece]:
    """
    Front + back sheath. If sleeve != none — append set-in sleeve piece.
    Geometry: EMKO grid + shared armscye splines.
    """
    fig = figure or FigureOptions()
    # Apply ease from spec onto a working copy of measurements
    m_work = Measurements(
        OG=m.OG,
        OT=m.OT,
        OB=m.OB,
        DTS=m.DTS,
        DTP=m.DTP,
        DI=resolve_length_cm(spec, fallback_di=m.DI),
        VPK=m.VPK,
        ShP=m.ShP,
        Vg=m.Vg,
        Cg=m.Cg,
        Pruh=spec.ease.bust_cm if spec.ease.bust_cm > 0 else m.Pruh,
        Ptal=spec.ease.waist_cm if spec.ease.waist_cm > 0 else m.Ptal,
        Pbed=spec.ease.hip_cm if spec.ease.hip_cm > 0 else m.Pbed,
        sleeve_len=m.sleeve_len,
        sleeve_w=m.sleeve_w,
        pr_len=m.pr_len,
    )
    grid = calculate_grid(m_work, fig)
    back = _draft_back(m_work, grid, fig, spec)
    front = _draft_front(m_work, grid, fig, spec)
    pieces = [back, front]
    if spec.sleeve != SleeveStyle.NONE:
        from atelier_cad.models.sleeve import draft_sleeve_set_in

        pieces.extend(draft_sleeve_set_in(m_work, spec, fig))
    return pieces


def _draft_back(m: Measurements, grid: dict, opts: FigureOptions, spec: PatternSpec) -> PatternPiece:
    W, L, D = grid["W"], grid["L"], grid["D"]
    o = opts.to_dict()
    neck_w = (m.OG / 13) + 2.5
    neck_h = neck_w / 3

    # Neckline (back): boat / round / jewel — control points ближе к ЕМКО
    if spec.neckline == Neckline.BOAT:
        neck_w = neck_w + 1.5
        neck_h = max(1.0, neck_h * 0.55)
        neck_pts = quadratic_bezier(
            (0.0, 0.0),
            (neck_w * 0.62, -neck_h * 0.08),
            (neck_w, -neck_h),
            n=14,
        )
    elif spec.neckline == Neckline.JEWEL:
        neck_pts = quadratic_bezier(
            (0.0, 0.0),
            (neck_w * 0.35, -neck_h * 0.05),
            (neck_w, -neck_h * 0.85),
            n=14,
        )
    else:
        neck_pts = quadratic_bezier(
            (0.0, 0.0),
            (neck_w * 0.45, -neck_h * 0.12),
            (neck_w, -neck_h),
            n=14,
        )

    angle = 15 + (5 if o.get("shoulder") == "покатые" else -5 if o.get("shoulder") == "прямые" else 0)
    rad = np.radians(angle)
    dart_val = 1.5 if o.get("posture") != "сутулая" else 2.5
    sh_len = m.ShP + dart_val
    sh_x = neck_w + sh_len * np.cos(rad)
    sh_y = neck_pts[-1][1] + sh_len * np.sin(rad)

    center_arm = W["back"] + W["arm"] / 2
    armscye = back_armscye_curve(
        (float(sh_x), float(sh_y)),
        w_back=W["back"],
        w_arm=W["arm"],
        depth_g=L["G"],
        n=22,
    )

    side_val = D["side"] / 2
    waist_x = center_arm - side_val
    hip_excess = (m.OB / 2 + m.Pbed / 2) - W["total"]
    hip_sh = hip_excess / 2
    hip_x = waist_x - hip_sh
    hem_x = hip_x  # fitted sheath

    sew: list[tuple[float, float]] = []
    sew.extend(neck_pts)
    sew.append((sh_x, float(sh_y)))
    sew.extend(armscye[1:])
    sew.append((waist_x, L["T"]))
    sew.append((hip_x, L["B"]))
    sew.append((hem_x, L["N"]))
    sew.append((0.0, L["N"]))
    sew.append((0.0, 0.0))

    # Shoulder dart (schematic triangle along shoulder)
    d_start = neck_w + 4.0
    dart = [
        (d_start, float(neck_pts[-1][1] + (d_start - neck_w) * np.sin(rad))),
        (d_start + 0.75, float(neck_pts[-1][1] + 8)),
        (d_start + dart_val, float(neck_pts[-1][1] + (d_start - neck_w + dart_val) * np.sin(rad))),
    ]

    cut = apply_seam_allowance(sew, spec.seam_allowance_cm)
    return PatternPiece(
        name="Спинка платья",
        cut_outline=cut,
        sew_outline=sew,
        darts=[dart],
        notches=[(center_arm, L["G"]), (waist_x, L["T"]), (hip_x, L["B"])],
        grainline=((1.5, 2.0), (1.5, L["N"] - 2.0)),
        labels=[((W["back"] * 0.35, L["T"]), "Спинка"), ((0.6, 2.0), "СС")],
        seam_allowance_cm=spec.seam_allowance_cm,
        hem_allowance_cm=spec.hem_allowance_cm,
    )


def _draft_front(m: Measurements, grid: dict, opts: FigureOptions, spec: PatternSpec) -> PatternPiece:
    W, L, D = grid["W"], grid["L"], grid["D"]
    Misc = grid["Misc"]
    bal = m.DTP - m.DTS
    start_y = -bal
    # Draft front in its own coordinate system (CF at x=0, side positive)
    # Width of front panel ≈ W['front'] + half armhole share to side seam
    front_w = W["front"] + W["arm"] / 2

    neck_w = (m.OG / 13) + 3.0
    if spec.neckline == Neckline.BOAT:
        neck_depth = 1.2
        neck_w = neck_w + 2.0
        neck_pts = quadratic_bezier(
            (0.0, start_y),
            (neck_w * 0.55, start_y + neck_depth * 0.2),
            (neck_w, start_y + neck_depth),
            n=14,
        )
    elif spec.neckline == Neckline.V:
        neck_depth = neck_w + 3.0
        neck_pts = quadratic_bezier(
            (0.0, start_y),
            (neck_w * 0.12, start_y + neck_depth * 0.92),
            (neck_w, start_y + neck_depth * 0.38),
            n=16,
        )
    elif spec.neckline == Neckline.SQUARE:
        neck_depth = neck_w * 0.7
        neck_pts = [
            (0.0, start_y),
            (0.0, start_y + neck_depth),
            (neck_w * 0.92, start_y + neck_depth),
            (neck_w, start_y + neck_depth * 0.35),
        ]
    else:
        neck_depth = neck_w + 1.5
        neck_pts = quadratic_bezier(
            (0.0, start_y),
            (neck_w * 0.18, start_y + neck_depth * 0.95),
            (neck_w, start_y + neck_depth * 0.42),
            n=16,
        )

    # Shoulder
    o = opts.to_dict()
    angle = 15 + (5 if o.get("shoulder") == "покатые" else -5 if o.get("shoulder") == "прямые" else 0)
    rad = np.radians(angle)
    sh_len = m.ShP
    sh_x = neck_pts[-1][0] + sh_len * np.cos(rad)
    sh_y = neck_pts[-1][1] + sh_len * np.sin(rad)

    underarm = (front_w, L["G"])
    armscye = front_armscye_curve(
        (float(sh_x), float(sh_y)),
        underarm,
        depth_g=L["G"],
        n=22,
    )

    side_val = D["side"] / 2
    waist_x = front_w - side_val
    hip_excess = (m.OB / 2 + m.Pbed / 2) - W["total"]
    hip_x = waist_x - hip_excess / 2
    hem_x = hip_x

    sew: list[tuple[float, float]] = []
    sew.extend([(float(x), float(y)) for x, y in neck_pts])
    sew.append((float(sh_x), float(sh_y)))
    sew.extend([(float(x), float(y)) for x, y in armscye[1:]])
    sew.append((waist_x, L["T"]))
    sew.append((hip_x, L["B"]))
    sew.append((hem_x, L["N"]))
    sew.append((0.0, L["N"]))
    sew.append((0.0, start_y))

    # Bust dart toward apex
    apex_x = m.Cg
    apex_y = start_y + m.Vg
    bust_dart = Misc["bust_dart"]
    dart = [
        (apex_x + 4.0, apex_y - bust_dart / 2),
        (apex_x, apex_y),
        (apex_x + 4.0, apex_y + bust_dart / 2),
    ]

    # Shift so min Y = 0 for layout friendliness
    min_y = min(y for _, y in sew)
    sew_shifted = [(x, y - min_y) for x, y in sew]
    dart_s = [(x, y - min_y) for x, y in dart]
    apex_y_s = apex_y - min_y

    cut = apply_seam_allowance(sew_shifted, spec.seam_allowance_cm)
    return PatternPiece(
        name="Перед платья",
        cut_outline=cut,
        sew_outline=sew_shifted,
        darts=[dart_s],
        notches=[
            (front_w, L["G"] - min_y),
            (waist_x, L["T"] - min_y),
            (hip_x, L["B"] - min_y),
        ],
        grainline=((1.5, 2.0), (1.5, L["N"] - min_y - 2.0)),
        labels=[
            ((front_w * 0.3, (L["T"] - min_y)), "Перед"),
            ((0.6, 2.0), "СН"),
            ((apex_x, apex_y_s), "●"),
        ],
        seam_allowance_cm=spec.seam_allowance_cm,
        hem_allowance_cm=spec.hem_allowance_cm,
    )
