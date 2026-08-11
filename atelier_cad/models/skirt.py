"""Straight and A-line skirt drafting (half-front + half-back)."""

from __future__ import annotations

from atelier_cad.geometry import PatternPiece, apply_seam_allowance
from atelier_cad.measurements import FigureOptions, Measurements
from atelier_cad.pattern_spec import GarmentType, PatternSpec, resolve_length_cm


def _skirt_length(m: Measurements, spec: PatternSpec) -> float:
    # For skirts DI is often entered as skirt length; resolve_length_cm prefers length_cm / preset
    return resolve_length_cm(spec, fallback_di=m.DI)


def draft_skirt(
    m: Measurements,
    spec: PatternSpec,
    figure: FigureOptions | None = None,
) -> list[PatternPiece]:
    """
    Classic half-panels: front and back, waist darts, side seam.
    Units cm. Y increases downward.
    """
    _ = figure  # reserved for posture adjustments
    length = _skirt_length(m, spec)
    waist_ease = spec.ease.waist_cm if spec.ease.waist_cm > 0 else m.Ptal
    hip_ease = spec.ease.hip_cm if spec.ease.hip_cm > 0 else m.Pbed

    # Full circumference with ease → half for front+back, then half-panel width
    waist_half = (m.OT + waist_ease) / 2
    hip_half = (m.OB + hip_ease) / 2

    # Hip depth from waist (classic ~18–20 cm)
    hip_depth = 18.0
    if length < hip_depth + 5:
        hip_depth = max(10.0, length * 0.35)

    # Dart distribution: ~2 darts on back, 1–2 on front (total intake)
    # Panel is half of half-circumference (quarter of body) for CF/CB cut-on-fold style:
    # We draft full front (CF to side) and full back (CB to side) = half of garment each.
    front_waist = waist_half / 2
    back_waist = waist_half / 2
    front_hip = hip_half / 2
    back_hip = hip_half / 2

    flare = 0.0
    if spec.garment_type == GarmentType.SKIRT_A_LINE or spec.silhouette.value == "a_line":
        flare = max(3.0, (hip_half - waist_half) * 0.35 + 4.0)

    seam = spec.seam_allowance_cm
    hem = spec.hem_allowance_cm

    front = _half_panel(
        name="Перед юбки",
        waist_w=front_waist,
        hip_w=front_hip,
        length=length,
        hip_depth=hip_depth,
        flare=flare,
        dart_intake=max(0.0, front_hip - front_waist) * 0.55,
        dart_from_cf=front_waist * 0.45,
        center_label="СН (сгиб)",
        seam=seam,
        hem=hem,
    )
    back = _half_panel(
        name="Спинка юбки",
        waist_w=back_waist,
        hip_w=back_hip,
        length=length,
        hip_depth=hip_depth,
        flare=flare * 0.9,
        dart_intake=max(0.0, back_hip - back_waist) * 0.7,
        dart_from_cf=back_waist * 0.4,
        center_label="СС (сгиб)",
        seam=seam,
        hem=hem,
        two_darts=True,
    )
    return [front, back]


def _half_panel(
    *,
    name: str,
    waist_w: float,
    hip_w: float,
    length: float,
    hip_depth: float,
    flare: float,
    dart_intake: float,
    dart_from_cf: float,
    center_label: str,
    seam: float,
    hem: float,
    two_darts: bool = False,
) -> PatternPiece:
    # Sew net (finished) outline: CF/CB at x=0, side at +width
    # Waist line y=0, hem y=length
    side_waist = waist_w
    side_hip = hip_w + flare * 0.15
    side_hem = hip_w + flare

    sew = [
        (0.0, 0.0),
        (side_waist, 0.0),
        (side_hip, hip_depth),
        (side_hem, length),
        (0.0, length),
    ]

    darts: list[list[tuple[float, float]]] = []
    if dart_intake > 0.4:
        if two_darts:
            half = dart_intake / 2
            for offset in (dart_from_cf * 0.7, dart_from_cf * 1.35):
                darts.append(_waist_dart(offset, half, depth=11.0))
        else:
            darts.append(_waist_dart(dart_from_cf, dart_intake, depth=10.0))

    cut = apply_seam_allowance(sew, seam)
    # Extend hem allowance beyond buffer: push bottom points of cut down by (hem - seam)
    # apply_seam_allowance already offsets all sides by seam; add extra hem if needed
    extra_hem = max(0.0, hem - seam)
    if extra_hem > 0:
        cut = [(x, y + extra_hem if y >= length - 0.01 else y) for x, y in cut]
        # Also expand bottom of sew reference for notches
        sew = [(x, y) for x, y in sew]
        sew[-2] = (sew[-2][0], sew[-2][1])  # side hem
        sew[-1] = (sew[-1][0], sew[-1][1])

    notches = [
        (side_hip, hip_depth),
        (side_waist / 2, 0.0),
    ]
    grain = ((2.0, 2.0), (2.0, length - 2.0))
    labels = [
        ((waist_w * 0.35, length * 0.45), name),
        ((0.8, 1.5), center_label),
        ((side_hip - 1.5, hip_depth), "бок"),
    ]

    return PatternPiece(
        name=name,
        cut_outline=cut,
        sew_outline=sew + [sew[0]],
        darts=darts,
        notches=notches,
        grainline=grain,
        labels=labels,
        seam_allowance_cm=seam,
        hem_allowance_cm=hem,
    )


def _waist_dart(center_x: float, intake: float, depth: float) -> list[tuple[float, float]]:
    half = intake / 2
    return [
        (center_x - half, 0.0),
        (center_x, depth),
        (center_x + half, 0.0),
    ]
