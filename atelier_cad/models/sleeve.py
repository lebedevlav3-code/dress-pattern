"""Set-in sleeve (втачной рукав) — отдельная деталь по сетке ЕМКО / длине проймы."""

from __future__ import annotations

import numpy as np

from atelier_cad.armscye import estimate_armscye_half_lengths, polyline_length
from atelier_cad.drafting import calculate_grid
from atelier_cad.geometry import PatternPiece, apply_seam_allowance, cubic_bezier
from atelier_cad.measurements import FigureOptions, Measurements
from atelier_cad.pattern_spec import PatternSpec, SleeveStyle


def _sleeve_length_cm(m: Measurements, spec: PatternSpec) -> float:
    if m.sleeve_len and m.sleeve_len >= 15:
        return float(m.sleeve_len)
    style = spec.sleeve
    if style == SleeveStyle.SHORT:
        return 18.0
    if style == SleeveStyle.THREE_QUARTER:
        return 42.0
    # LONG / SET_IN / default
    return max(50.0, m.ShP * 4.2)


def _sleeve_flat_width_cm(m: Measurements, armscye_total: float) -> float:
    """Ширина детали между подрезами (½ окружности рукава в плоскости)."""
    if m.sleeve_w and m.sleeve_w >= 20:
        # sleeve_w = окружность руки с прибавкой
        return float(m.sleeve_w) / 2
    # Эвристика: ширина ≈ половина длины проймы + небольшой запас
    return max(14.0, armscye_total / 2 + 1.2)


def draft_sleeve_set_in(
    m: Measurements,
    spec: PatternSpec,
    figure: FigureOptions | None = None,
) -> list[PatternPiece]:
    """
    Полный втачной рукав (одна деталь; qty=2 в PatternSpec — кроить дважды).

    - высота оката ≈ 0.38 × глубина проймы сетки
    - длина оката ≈ длина проймы переда+спинки + ease ~1.8 см
    """
    fig = figure or FigureOptions()
    m_work = Measurements(
        OG=m.OG,
        OT=m.OT,
        OB=m.OB,
        DTS=m.DTS,
        DTP=m.DTP,
        DI=m.DI,
        VPK=m.VPK,
        ShP=m.ShP,
        Vg=m.Vg,
        Cg=m.Cg,
        Pruh=spec.ease.bust_cm if spec.ease.bust_cm > 0 else m.Pruh,
        Ptal=m.Ptal,
        Pbed=m.Pbed,
        sleeve_len=m.sleeve_len,
        sleeve_w=m.sleeve_w,
        pr_len=m.pr_len,
    )
    grid = calculate_grid(m_work, fig)
    W, L = grid["W"], grid["L"]
    depth_g = L["G"]

    neck_w_b = (m_work.OG / 13) + 2.5
    rad = np.radians(15.0)
    sh_back = (
        float(neck_w_b + m_work.ShP * np.cos(rad)),
        float(-(neck_w_b / 3) + m_work.ShP * np.sin(rad)),
    )
    bal = m_work.DTP - m_work.DTS
    neck_w_f = (m_work.OG / 13) + 3.0
    sh_front = (
        float(neck_w_f + m_work.ShP * np.cos(rad)),
        float(-bal + neck_w_f * 0.35 + m_work.ShP * np.sin(rad)),
    )
    back_len, front_len, arm_total = estimate_armscye_half_lengths(
        w_back=W["back"],
        w_arm=W["arm"],
        w_front=W["front"],
        depth_g=depth_g,
        sh_back=sh_back,
        sh_front=sh_front,
    )

    cap_h = max(10.0, depth_g * 0.38)
    width = _sleeve_flat_width_cm(m_work, arm_total)
    sleeve_len = _sleeve_length_cm(m_work, spec)
    under_len = max(20.0, sleeve_len - cap_h)
    cx = width / 2

    back_share = back_len / arm_total if arm_total > 0 else 0.5
    front_share = 1.0 - back_share

    left_ua = (0.0, cap_h)
    right_ua = (width, cap_h)
    crown = (cx, 0.0)

    # Окат: слева доля спинки, справа — переда
    back_cap = cubic_bezier(
        left_ua,
        (cx * 0.22, cap_h * 0.58),
        (cx * (0.35 + 0.25 * back_share), cap_h * 0.14),
        crown,
        n=24,
    )
    front_cap = cubic_bezier(
        crown,
        (cx + (width - cx) * (0.35 + 0.2 * front_share), cap_h * 0.14),
        (cx + (width - cx) * 0.78, cap_h * 0.58),
        right_ua,
        n=24,
    )

    cap_len = polyline_length(back_cap) + polyline_length(front_cap[1:])

    hem_w = width * 0.72
    hem_left = (cx - hem_w / 2, cap_h + under_len)
    hem_right = (cx + hem_w / 2, cap_h + under_len)

    sew: list[tuple[float, float]] = []
    sew.extend(back_cap)
    sew.extend(front_cap[1:])
    sew.append(hem_right)
    sew.append(hem_left)
    sew.append(left_ua)

    back_pitch = back_cap[max(1, len(back_cap) // 3)]
    front_pitch = front_cap[max(1, (2 * len(front_cap)) // 3)]

    cut = apply_seam_allowance(sew, spec.seam_allowance_cm)

    qty = 2
    for p in spec.pieces:
        if p.role.value == "sleeve":
            qty = p.qty
            break

    return [
        PatternPiece(
            name=f"Рукав втачной ×{qty}",
            cut_outline=cut,
            sew_outline=sew,
            darts=[],
            notches=[crown, back_pitch, front_pitch, left_ua, right_ua],
            grainline=((cx, 2.0), (cx, cap_h + under_len - 2.0)),
            labels=[
                ((cx, cap_h + under_len * 0.45), "Рукав втачной"),
                ((cx, 1.2), "окат"),
                ((1.0, cap_h + 1.2), "спинка"),
                ((width - 3.0, cap_h + 1.2), "перед"),
                (
                    (cx, cap_h + under_len * 0.18),
                    f"окат≈{cap_len:.1f} / пройма≈{arm_total:.1f} см",
                ),
            ],
            seam_allowance_cm=spec.seam_allowance_cm,
            hem_allowance_cm=min(spec.hem_allowance_cm, 3.0),
        )
    ]
