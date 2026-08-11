"""Tiled A4 PDF export with 5×5 cm test square and overlap marks."""

from __future__ import annotations

import io
import math
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from atelier_cad.geometry import PatternPiece, union_bounds

# A4 in inches
A4_W_IN = 8.27
A4_H_IN = 11.69
# Printable margins (printer safe area)
MARGIN_IN = 0.4
# Overlap between tiles for glue alignment (cm)
OVERLAP_CM = 1.0
# Test square side (cm)
TEST_SQUARE_CM = 5.0


def _draw_pieces(ax, pieces: Sequence[PatternPiece]) -> None:
    for piece in pieces:
        xs = [p[0] for p in piece.cut_outline]
        ys = [p[1] for p in piece.cut_outline]
        ax.plot(xs, ys, "k-", lw=1.2, solid_capstyle="round")
        if piece.sew_outline:
            sxs = [p[0] for p in piece.sew_outline]
            sys = [p[1] for p in piece.sew_outline]
            ax.plot(sxs, sys, color="#3366cc", lw=0.8, ls="--")
        for dart in piece.darts:
            ax.plot([p[0] for p in dart], [p[1] for p in dart], color="#3366cc", lw=0.7)
        if piece.grainline:
            (x1, y1), (x2, y2) = piece.grainline
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="<->", color="#666", lw=0.7),
            )
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 - 0.6, "ДОЛЕВАЯ", fontsize=6, color="#666", ha="center")
        for nx, ny in piece.notches:
            ax.plot([nx - 0.5, nx + 0.5], [ny, ny], color="#c00", lw=1.0)
        for (x, y), text in piece.labels:
            ax.text(x, y, text, fontsize=7, color="#222")


def _draw_test_square(ax, origin: tuple[float, float] = (0.5, 0.5)) -> None:
    x0, y0 = origin
    ax.add_patch(
        Rectangle(
            (x0, y0),
            TEST_SQUARE_CM,
            TEST_SQUARE_CM,
            fill=False,
            edgecolor="red",
            linewidth=1.2,
        )
    )
    ax.text(
        x0 + TEST_SQUARE_CM / 2,
        y0 + TEST_SQUARE_CM / 2,
        "5×5 см\nпроверь\nмасштаб",
        ha="center",
        va="center",
        fontsize=7,
        color="red",
    )


def _draw_overlap_marks(ax, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    """Corner L-marks for aligning tiled pages."""
    mark = 1.0  # cm
    color = "#cc0000"
    # four corners
    corners = [
        (x_min, y_min, 1, 1),
        (x_max, y_min, -1, 1),
        (x_min, y_max, 1, -1),
        (x_max, y_max, -1, -1),
    ]
    for cx, cy, sx, sy in corners:
        ax.plot([cx, cx + sx * mark], [cy, cy], color=color, lw=0.9)
        ax.plot([cx, cx], [cy, cy + sy * mark], color=color, lw=0.9)


def render_preview_figure(pieces: Sequence[PatternPiece]) -> plt.Figure:
    width_cm, height_cm = union_bounds(pieces, padding=3.0)
    fig_w = max(6.0, min(12.0, width_cm / 8))
    fig_h = max(6.0, min(16.0, height_cm / 8))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _draw_pieces(ax, pieces)
    _draw_test_square(ax)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title("Atelier CAD — превью", fontsize=10)
    return fig


def save_tiled_pdf(
    pieces: Sequence[PatternPiece],
    *,
    title: str = "Atelier CAD",
) -> bytes:
    """
    Slice pattern into A4 pages with overlap marks and a 5×5 cm test square
    on the first page. Scale is absolute: 1 plot unit = 1 cm.
    """
    width_cm, height_cm = union_bounds(pieces, padding=2.0)
    # Ensure room for test square on first tile
    width_cm = max(width_cm, TEST_SQUARE_CM + 4)
    height_cm = max(height_cm, TEST_SQUARE_CM + 4)

    work_w_in = A4_W_IN - 2 * MARGIN_IN
    work_h_in = A4_H_IN - 2 * MARGIN_IN
    work_w_cm = work_w_in * 2.54 - OVERLAP_CM
    work_h_cm = work_h_in * 2.54 - OVERLAP_CM

    cols = max(1, math.ceil(width_cm / work_w_cm))
    rows = max(1, math.ceil(height_cm / work_h_cm))

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Cover / instruction page
        fig_cover, ax_c = plt.subplots(figsize=(A4_W_IN, A4_H_IN))
        ax_c.axis("off")
        ax_c.text(0.5, 0.85, title, ha="center", fontsize=16, transform=ax_c.transAxes)
        ax_c.text(
            0.5,
            0.72,
            "Инструкция по склейке\n"
            "1) Распечатайте без подгонки масштаба (100% / Actual size).\n"
            "2) Измерьте красный квадрат 5×5 см на 1-м листе чертежа.\n"
            "3) Совместите угловые метки соседних листов, склейте внахлёст.\n"
            "4) Проверьте контрольные мерки на макете (±1–1.5 см).",
            ha="center",
            va="top",
            fontsize=11,
            transform=ax_c.transAxes,
        )
        ax_c.text(
            0.5,
            0.35,
            f"Листов чертежа: {rows}×{cols} = {rows * cols}\n"
            f"Габарит: {width_cm:.1f} × {height_cm:.1f} см\n"
            f"Overlap: {OVERLAP_CM:.1f} см",
            ha="center",
            fontsize=10,
            transform=ax_c.transAxes,
        )
        ax_c.text(
            0.5,
            0.12,
            "Посадка проверяется на макете. PDF — не обещание идеальной посадки с первого раза.",
            ha="center",
            fontsize=8,
            color="#666",
            transform=ax_c.transAxes,
        )
        pdf.savefig(fig_cover)
        plt.close(fig_cover)

        for r in range(rows):
            for c in range(cols):
                x_min = c * work_w_cm
                y_min = r * work_h_cm
                x_max = x_min + work_w_cm + OVERLAP_CM
                y_max = y_min + work_h_cm + OVERLAP_CM

                fig, ax = plt.subplots(figsize=(A4_W_IN, A4_H_IN))
                # Leave margin in figure coordinates via subplot adjust
                fig.subplots_adjust(
                    left=MARGIN_IN / A4_W_IN,
                    right=1 - MARGIN_IN / A4_W_IN,
                    bottom=MARGIN_IN / A4_H_IN,
                    top=1 - MARGIN_IN / A4_H_IN,
                )
                _draw_pieces(ax, pieces)
                if r == 0 and c == 0:
                    _draw_test_square(ax, origin=(x_min + 0.8, y_min + 0.8))
                _draw_overlap_marks(ax, x_min, x_max, y_min, y_max)
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_max, y_min)  # Y down
                ax.set_aspect("equal")
                ax.set_title(f"Лист {r + 1}-{c + 1}  (ряд {r + 1}, кол {c + 1})", fontsize=9, color="#a00")
                # Light frame
                ax.add_patch(
                    Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        fill=False,
                        edgecolor="#dddddd",
                        linewidth=0.5,
                    )
                )
                pdf.savefig(fig)
                plt.close(fig)

    buf.seek(0)
    return buf.read()


# ISO A0 in inches (portrait)
A0_W_IN = 33.1
A0_H_IN = 46.8


def save_a0_pdf(
    pieces: Sequence[PatternPiece],
    *,
    title: str = "Atelier CAD",
) -> bytes:
    """Single-sheet A0 PDF at true scale (1 unit = 1 cm) with 5×5 cm test square."""
    width_cm, height_cm = union_bounds(pieces, padding=3.0)
    width_cm = max(width_cm, TEST_SQUARE_CM + 6)
    height_cm = max(height_cm, TEST_SQUARE_CM + 6)

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig, ax = plt.subplots(figsize=(A0_W_IN, A0_H_IN))
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.95)
        _draw_pieces(ax, pieces)
        _draw_test_square(ax, origin=(1.0, 1.0))
        ax.set_xlim(-1, max(width_cm, 1))
        ax.set_ylim(max(height_cm, 1), -1)
        ax.set_aspect("equal")
        ax.set_title(f"{title} — лист A0 (печать 100% / Actual size)", fontsize=14)
        ax.text(
            0.5,
            -0.02,
            "Проверьте красный квадрат 5×5 см. Посадка — на макете.",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            color="#666",
        )
        pdf.savefig(fig)
        plt.close(fig)

    buf.seek(0)
    return buf.read()
