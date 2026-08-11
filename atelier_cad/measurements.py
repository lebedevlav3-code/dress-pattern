"""Body measurements and ease (прибавки). Units: centimetres."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class Measurements:
    """Client body measurements + ease. All values in cm."""

    OG: float  # bust circumference
    OT: float  # waist
    OB: float  # hips
    DTS: float  # back length to waist
    DTP: float  # front length to waist
    DI: float  # garment length (from neck/waist depending on model)
    VPK: float  # back shoulder height
    ShP: float  # shoulder width
    Vg: float  # bust height
    Cg: float  # bust centre distance (half)
    Pruh: float  # bust ease
    Ptal: float  # waist ease
    Pbed: float  # hip ease
    # Sleeve (optional for bodice)
    pr_len: float = 0.0
    sleeve_len: float = 0.0
    sleeve_w: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Measurements:
        known = {f.name for f in fields(cls)}
        filtered = {k: float(v) for k, v in data.items() if k in known}
        return cls(**filtered)

    def validate(self) -> list[str]:
        """Return list of human-readable validation errors (RU)."""
        errors: list[str] = []
        if self.OG < 70 or self.OG > 150:
            errors.append("ОГ вне разумного диапазона (70–150 см)")
        if self.OT < 50 or self.OT > 140:
            errors.append("ОТ вне разумного диапазона (50–140 см)")
        if self.OB < 70 or self.OB > 160:
            errors.append("ОБ вне разумного диапазона (70–160 см)")
        if self.OT > self.OG + 15:
            errors.append("ОТ заметно больше ОГ — проверьте мерки")
        if self.DI < 20 or self.DI > 180:
            errors.append("ДИ вне разумного диапазона (20–180 см)")
        if self.DTS < 30 or self.DTS > 55:
            errors.append("ДТС вне разумного диапазона (30–55 см)")
        if self.DTP < 30 or self.DTP > 65:
            errors.append("ДТП вне разумного диапазона (30–65 см)")
        return errors


@dataclass
class FigureOptions:
    """Figure options that adjust drafting rules (not body circumferences)."""

    bust: str = "средняя"  # полная | средняя | маленькая
    shoulder: str = "нормальные"  # покатые | нормальные | прямые
    posture: str = "нормальная"  # сутулая | нормальная | перегибистая

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


DEFAULT_MEASUREMENTS = Measurements(
    OG=96.0,
    OT=76.0,
    OB=104.0,
    DTS=42.0,
    DTP=44.0,
    DI=60.0,
    VPK=42.0,
    ShP=13.0,
    Vg=27.0,
    Cg=20.0,
    Pruh=4.0,
    Ptal=2.0,
    Pbed=2.0,
)
