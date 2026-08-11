"""PatternSpec — structured parameters for parametric drafting (NOT raw SVG)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class PatternSpecError(ValueError):
    """Raised when PatternSpec is invalid — callers must not emit a broken PDF."""


class GarmentType(str, Enum):
    SKIRT_STRAIGHT = "skirt_straight"
    SKIRT_A_LINE = "skirt_a_line"
    DRESS_SHEATH = "dress_sheath"
    SLEEVE_SET_IN = "sleeve_set_in"
    T_SHIRT = "t_shirt"


class Silhouette(str, Enum):
    FITTED = "fitted"
    SEMI = "semi"
    RELAXED = "relaxed"
    A_LINE = "a_line"
    STRAIGHT = "straight"


class LengthPreset(str, Enum):
    MINI = "mini"
    ABOVE_KNEE = "above_knee"
    KNEE = "knee"
    MIDI = "midi"
    TEA = "tea"
    MAXI = "maxi"
    CUSTOM = "custom"


class Neckline(str, Enum):
    ROUND = "round"
    BOAT = "boat"  # лодочка
    V = "v"
    SQUARE = "square"
    JEWEL = "jewel"
    NONE = "none"  # skirts


class SleeveStyle(str, Enum):
    NONE = "none"
    SET_IN = "set_in"
    SHORT = "short"
    THREE_QUARTER = "three_quarter"
    LONG = "long"


class PieceRole(str, Enum):
    FRONT = "front"
    BACK = "back"
    SLEEVE = "sleeve"
    WAISTBAND = "waistband"
    FACING = "facing"
    OTHER = "other"


class EaseSpec(BaseModel):
    bust_cm: float = Field(0.0, ge=0, le=20)
    waist_cm: float = Field(0.0, ge=0, le=20)
    hip_cm: float = Field(0.0, ge=0, le=20)


class PieceSpec(BaseModel):
    role: PieceRole
    name_ru: str
    qty: int = Field(1, ge=1, le=8)


class PatternSpec(BaseModel):
    """Parameters that drive parametric templates. LLM fills this; CAD builds geometry."""

    garment_type: GarmentType
    silhouette: Silhouette = Silhouette.STRAIGHT
    length: LengthPreset = LengthPreset.MIDI
    length_cm: float | None = Field(
        default=None,
        description="Absolute garment length in cm when length=CUSTOM or override",
        ge=15,
        le=180,
    )
    neckline: Neckline = Neckline.NONE
    sleeve: SleeveStyle = SleeveStyle.NONE
    reliefs: bool = False
    pockets: bool = False
    waistband_cm: float = Field(0.0, ge=0, le=12)
    seam_allowance_cm: float = Field(1.5, ge=0.5, le=3.0)
    hem_allowance_cm: float = Field(4.0, ge=1.0, le=10.0)
    ease: EaseSpec = Field(default_factory=EaseSpec)
    pieces: list[PieceSpec] = Field(default_factory=list)
    notes: str = ""
    source: Literal["manual", "openrouter", "vision", "preset"] = "manual"

    @field_validator("notes")
    @classmethod
    def notes_not_too_long(cls, v: str) -> str:
        if len(v) > 2000:
            raise ValueError("notes too long")
        return v

    @model_validator(mode="after")
    def default_pieces_and_consistency(self) -> PatternSpec:
        if self.garment_type in (GarmentType.SKIRT_STRAIGHT, GarmentType.SKIRT_A_LINE):
            if self.neckline not in (Neckline.NONE,):
                # skirts have no neckline — coerce
                object.__setattr__(self, "neckline", Neckline.NONE)
            if self.sleeve != SleeveStyle.NONE:
                object.__setattr__(self, "sleeve", SleeveStyle.NONE)
            if not self.pieces:
                object.__setattr__(
                    self,
                    "pieces",
                    [
                        PieceSpec(role=PieceRole.FRONT, name_ru="Перед юбки", qty=1),
                        PieceSpec(role=PieceRole.BACK, name_ru="Спинка юбки", qty=1),
                    ],
                )
            if self.garment_type == GarmentType.SKIRT_A_LINE and self.silhouette == Silhouette.STRAIGHT:
                object.__setattr__(self, "silhouette", Silhouette.A_LINE)

        if self.garment_type == GarmentType.DRESS_SHEATH:
            if self.neckline == Neckline.NONE:
                object.__setattr__(self, "neckline", Neckline.BOAT)
            if not self.pieces:
                pieces = [
                    PieceSpec(role=PieceRole.FRONT, name_ru="Перед платья", qty=1),
                    PieceSpec(role=PieceRole.BACK, name_ru="Спинка платья", qty=1),
                ]
                if self.sleeve != SleeveStyle.NONE:
                    pieces.append(
                        PieceSpec(role=PieceRole.SLEEVE, name_ru="Рукав втачной", qty=2)
                    )
                object.__setattr__(self, "pieces", pieces)
            if self.sleeve == SleeveStyle.NONE:
                pass  # sleeveless sheath is valid

        if self.garment_type == GarmentType.SLEEVE_SET_IN:
            if self.sleeve == SleeveStyle.NONE:
                object.__setattr__(self, "sleeve", SleeveStyle.SET_IN)
            if self.neckline != Neckline.NONE:
                object.__setattr__(self, "neckline", Neckline.NONE)
            if not self.pieces:
                object.__setattr__(
                    self,
                    "pieces",
                    [PieceSpec(role=PieceRole.SLEEVE, name_ru="Рукав втачной", qty=2)],
                )
        return self

    def to_public_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def parse_pattern_spec(data: dict[str, Any] | PatternSpec) -> PatternSpec:
    """Validate raw dict → PatternSpec or raise PatternSpecError."""
    if isinstance(data, PatternSpec):
        return data
    try:
        return PatternSpec.model_validate(data)
    except ValidationError as exc:
        raise PatternSpecError(f"Некорректный PatternSpec: {exc}") from exc


def preset_skirt_straight(*, length: LengthPreset = LengthPreset.MIDI) -> PatternSpec:
    return PatternSpec(
        garment_type=GarmentType.SKIRT_STRAIGHT,
        silhouette=Silhouette.STRAIGHT,
        length=length,
        neckline=Neckline.NONE,
        sleeve=SleeveStyle.NONE,
        ease=EaseSpec(waist_cm=2.0, hip_cm=2.0),
        source="preset",
        notes="Прямая юбка, пресет",
    )


def preset_skirt_a_line(*, length: LengthPreset = LengthPreset.MIDI) -> PatternSpec:
    return PatternSpec(
        garment_type=GarmentType.SKIRT_A_LINE,
        silhouette=Silhouette.A_LINE,
        length=length,
        ease=EaseSpec(waist_cm=2.0, hip_cm=4.0),
        source="preset",
        notes="Юбка-трапеция, пресет",
    )


def preset_dress_sheath_boat_midi() -> PatternSpec:
    return PatternSpec(
        garment_type=GarmentType.DRESS_SHEATH,
        silhouette=Silhouette.FITTED,
        length=LengthPreset.MIDI,
        neckline=Neckline.BOAT,
        sleeve=SleeveStyle.NONE,
        ease=EaseSpec(bust_cm=3.0, waist_cm=2.0, hip_cm=2.0),
        source="preset",
        notes="Платье-футляр миди, лодочка",
    )


def preset_sleeve_set_in_long() -> PatternSpec:
    return PatternSpec(
        garment_type=GarmentType.SLEEVE_SET_IN,
        silhouette=Silhouette.SEMI,
        length=LengthPreset.CUSTOM,
        length_cm=58.0,
        neckline=Neckline.NONE,
        sleeve=SleeveStyle.LONG,
        ease=EaseSpec(bust_cm=3.0),
        source="preset",
        notes="Втачной рукав длинный (отдельная деталь)",
    )


def preset_dress_sheath_with_sleeve() -> PatternSpec:
    return PatternSpec(
        garment_type=GarmentType.DRESS_SHEATH,
        silhouette=Silhouette.FITTED,
        length=LengthPreset.MIDI,
        neckline=Neckline.ROUND,
        sleeve=SleeveStyle.LONG,
        ease=EaseSpec(bust_cm=3.0, waist_cm=2.0, hip_cm=2.0),
        source="preset",
        notes="Платье-футляр миди + втачной рукав",
    )


LENGTH_CM_DEFAULTS: dict[LengthPreset, float] = {
    LengthPreset.MINI: 40.0,
    LengthPreset.ABOVE_KNEE: 50.0,
    LengthPreset.KNEE: 58.0,
    LengthPreset.MIDI: 70.0,
    LengthPreset.TEA: 90.0,
    LengthPreset.MAXI: 110.0,
    LengthPreset.CUSTOM: 70.0,
}


def resolve_length_cm(spec: PatternSpec, fallback_di: float | None = None) -> float:
    if spec.length_cm is not None:
        return float(spec.length_cm)
    if fallback_di is not None and fallback_di > 0:
        return float(fallback_di)
    return LENGTH_CM_DEFAULTS.get(spec.length, 70.0)
