# app_full_dress_figure.py — конструктор платья и рукава
# с учётом индивидуальных особенностей фигуры

import streamlit as st
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np
import io

@dataclass
class Measurements:
    OG: float
    OT: float
    OB: float
    DTS: float
    DTP: float
    DI: float
    VPK: float
    ShP: float
    Pruh: float
    Ptal: float
    Pbed: float
    pr_armhole_len: float
    sleeve_len: float
    sleeve_width_bottom: float

def calc_base_grid(m: Measurements):
    width_g = (m.OG + m.Pruh) / 2
    width_t = (m.OT + m.Ptal) / 2
    width_b = (m.OB + m.Pbed) / 2
    params = {
        "Ширина груди": width_g,
        "Ширина талии": width_t,
        "Ширина бедер": width_b,
        "Зона спинки": 20.5,
        "Зона проймы": 14.0,
        "Зона переда": width_g - 20.5 - 14.0,
        "Глубина проймы": 23.5,
        "ДТС": m.DTS,
        "ДТП": m.DTP
    }
    return params

def calc_sleeve(m: Measurements):
    return {
        "Высота оката": m.pr_armhole_len / 3,
        "Ширина рукава": (m.OG / 3) + 3
    }

# ===== КОРРЕКТИРОВКИ ПО ФИГУРЕ =====

def apply_figure_adjustments_body(params, options):
    """Корректировки корпуса по фигуре."""
    # копия, чтобы не портить оригинал
    p = params.copy()

    # ПЛЕЧИ
    if options["shoulder"] == "покатые":
        # углубляем пройму и усиливаем наклон плеча (используем как коэффициенты)
        p["Глубина проймы"] += 1.0      # +1 см глубже пройма
    elif options["shoulder"] == "прямые":
        p["Глубина проймы"] -= 0.5      # чуть мельче

    # ОСАНКА
    # сутулая: длина спины больше, перед чуть короче
    if options["posture"] == "сутулая":
        p["ДТС"] += 1.0
        p["ДТП"] -= 0.5
    # перегибистая: наоборот
    elif options["posture"] == "перегибистая":
        p["ДТС"] -= 0.5
        p["ДТП"] += 1.0

    # ГРУДЬ
    if options["bust"] == "полная":
        # расширяем перед и увеличиваем раствор нагрудной вытачки (абстрактный параметр)
        p["Зона переда"] += 1.0
        p["Ширина груди"] += 0.5
    elif options["bust"] == "малая":
        p["Зона переда"] -= 0.5

    # БЁДРА
    if options["hips"] == "полные":
        p["Ширина бедер"] += 1.5
    elif options["hips"] == "плоские":
        p["Ширина бедер"] -= 1.0

    # РОСТ (смещение талии и низа)
    if options["height"] == "ниже среднего":
        p["ДТС"] -= 1.0
        p["ДТП"] -= 1.0
    elif options["height"] == "выше среднего":
        p["ДТС"] += 1.0
        p["ДТП"] += 1.0

    return p

def apply_figure_adjustments_sleeve(sleeve_params, body_params, options):
    """Корректировки рукава по фигуре и глубине проймы."""
    s = sleeve_params.copy()

    # если пройма углублена (покатые плечи, сутулость) — немного увеличиваем высоту оката
    base_depth = 23.5
    delta_pr = body_params["Глубина проймы"] - base_depth
    s["Высота оката"] += delta_pr * 0.4  # не напрямую 1:1, а мягкий коэффициент

    # полная/малая грудь — влияние на свободу по окату
    if options["bust"] == "полная":
        s["Ширина рукава"] += 1.0
    elif options["bust"] == "малая":
        s["Ширина рукава"] -= 0.5

    # полные/плоские бёдра на рукав почти не влияют — можно игнорировать или чуть расширять низ
    if options["hips"] == "полные":
        s["Ширина рукава"] += 0.5

    return s

# ===== ПОСТРОЕНИЕ КОРПУСА =====

def plot_body(params, m, options):
    fig, ax = plt.subplots(figsize=(7, 11))
    total_w = params['Ширина груди']
    spine_x = params['Зона спинки']
    arm_x = spine_x + params['Зона проймы']
    G_y = params['Глубина проймы']
    T_y = params['ДТС']
    B_y = params['ДТС'] + 19
    N_y = m.DI

    # сетка
    for label, y in {'А':0, 'Г':G_y, 'Т':T_y, 'Б':B_y, 'Н':N_y}.items():
        ax.plot([0, total_w], [y, y], 'lightgray', lw=0.8)
        ax.text(-3, y, label, va='center', ha='right', fontsize=8)
    for x in [0, spine_x, arm_x, total_w]:
        ax.plot([x, x], [0, N_y], 'k--', lw=0.8)

    # горловина спинки
    nx_b, ny_b = 6.7, 2
    x_b = np.linspace(0, nx_b, 10)
    y_b = -ny_b * (x_b/nx_b)**2
    ax.plot(x_b, y_b, 'b', lw=1.8, label='Горловина спинки')

    # плечо спинки (покатость слегка усиливаем/уменьшаем по опции плеч)
    base_drop = m.VPK / 10
    if options["shoulder"] == "покатые":
        drop = base_drop + 0.7
    elif options["shoulder"] == "прямые":
        drop = base_drop - 0.5
    else:
        drop = base_drop
    px_b = nx_b + m.ShP * np.cos(np.deg2rad(12))
    py_b = -drop
    ax.plot([nx_b, px_b], [0, py_b], 'b', lw=1.4)

    # горловина переда
    nx_f, ny_f = 6.7, 7.7
    start = total_w
    x_f = np.linspace(start - nx_f, start, 10)
    y_f = -ny_f * (1 - (x_f - (start - nx_f)) / nx_f)**2 - 0.5
    ax.plot(x_f, y_f, 'm', lw=1.8, label='Горловина переда')

    # плечо переда (наклон немного корректируем аналогично)
    sx_f = start - 5
    ex_f = start - 5 - (m.ShP - 5)
    front_drop = -4.5
    if options["shoulder"] == "покатые":
        front_drop -= 0.5
    elif options["shoulder"] == "прямые":
        front_drop += 0.3
    ax.plot([sx_f, ex_f], [0, front_drop], 'm', lw=1.4)

    # проймы (кривые между плечом и линией груди)
    spine_arm = np.array([[px_b, py_b], [spine_x + 2, G_y - 10], [spine_x + 3, G_y]])
    front_arm = np.array([[ex_f, front_drop], [arm_x - 2, G_y - 11], [arm_x - 1, G_y]])
    ax.plot(spine_arm[:,0], spine_arm[:,1], 'b')
    ax.plot(front_arm[:,0], front_arm[:,1], 'm')

    ax.set_ylim(N_y + 5, -15)
    ax.set_xlim(-10, total_w + 10)
    ax.set_aspect("equal")
    ax.set_title("Базовая сетка платья с корректировками фигуры")
    ax.legend(fontsize=7)
    return fig

# ===== ПОСТРОЕНИЕ РУКАВА =====

def plot_sleeve(m, sleeve_params):
    fig, ax = plt.subplots(figsize=(6, 10))
    vok = sleeve_params['Высота оката']
    shr = sleeve_params['Ширина рукава']
    h = m.sleeve_len
    w = shr / 2

    # оси
    ax.plot([-w, w], [0, 0], 'gray', lw=0.8)
    ax.plot([0, 0], [0, h], 'gray', lw=0.8)

    # окат (упрощённая плавная кривая)
    x_left = np.linspace(-w, 0, 30)
    x_right = np.linspace(0, w, 30)
    y_left = -((x_left + w)**2)/(2*w) * (vok / w)
    y_right = -((x_right - w)**2)/(2*w) * (vok / w)
    ax.plot(np.concatenate([x_left, x_right]), np.concatenate([y_left, y_right]), 'b', lw=1.8)

    # низ рукава
    ax.plot([-m.sleeve_width_bottom/2, m.sleeve_width_bottom/2], [h, h], 'k', lw=1.2)

    # боковые швы
    ax.plot([-w, -m.sleeve_width_bottom/2], [0, h], 'k', lw=1.2)
    ax.plot([w, m.sleeve_width_bottom/2], [0, h], 'k', lw=1.2)

    ax.set_xlim(-w - 5, w + 5)
    ax.set_ylim(h + 10, -vok - 5)
    ax.set_aspect('equal')
    ax.set_title("Рукав с учётом корректировок проймы")
    ax.set_xlabel("см")
    ax.set_ylabel("длина рукава, см")
    return fig

# ====== UI ======

st.set_page_config(layout="wide")
st.title("🧵 Конструктор платья и рукава с учётом особенностей фигуры")

tab1, tab2 = st.tabs(["👗 Платье (корпус)", "👕 Рукав"])

with tab1:
    st.subheader("Мерки корпуса")
    OG = st.number_input("ОГ", 70.0, 130.0, 103.0)
    OT = st.number_input("ОТ", 60.0, 110.0, 86.0)
    OB = st.number_input("ОБ", 70.0, 130.0, 102.0)
    DTS = st.number_input("ДТС", 35.0, 45.0, 41.0)
    DTP = st.number_input("ДТП", 40.0, 55.0, 46.0)
    DI = st.number_input("ДИ", 80.0, 120.0, 110.0)
    VPK = st.number_input("Впк", 35.0, 45.0, 41.0)
    ShP = st.number_input("Шп", 10.0, 20.0, 14.0)
    Pruh = st.number_input("Прибавка по груди", 0.0, 10.0, 5.0)
    Ptal = st.number_input("Прибавка по талии", 0.0, 10.0, 3.0)
    Pbed = st.number_input("Прибавка по бедрам", 0.0, 10.0, 2.0)
    pr_armhole_len = st.number_input("Длина проймы (по лекалу)", 40.0, 60.0, 48.0)

    st.subheader("Особенности фигуры")
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        shoulder = st.selectbox("Плечи", ["нормальные", "покатые", "прямые"])
        posture = st.selectbox("Осанка", ["нормальная", "сутулая", "перегибистая"])
    with col_opt2:
        bust = st.selectbox("Грудь", ["средняя", "малая", "полная"])
        hips = st.selectbox("Бёдра", ["нормальные", "плоские", "полные"])
    with col_opt3:
        height = st.selectbox("Рост", ["средний", "ниже среднего", "выше среднего"])

    figure_options = {
        "shoulder": shoulder,
        "posture": posture,
        "bust": bust,
        "hips": hips,
        "height": height
    }

    if st.button("📐 Построить корпус с учётом фигуры"):
        m_body = Measurements(OG, OT, OB, DTS, DTP, DI, VPK, ShP, Pruh, Ptal, Pbed, pr_armhole_len, 60, 26)
        base_raw = calc_base_grid(m_body)
        base = apply_figure_adjustments_body(base_raw, figure_options)
        fig_body = plot_body(base, m_body, figure_options)
        st.pyplot(fig_body)

        pdf_buf, svg_buf = io.BytesIO(), io.BytesIO()
        fig_body.savefig(pdf_buf, format="pdf", bbox_inches="tight")
        fig_body.savefig(svg_buf, format="svg", bbox_inches="tight")
        st.download_button("📄 Скачать корпус (PDF)", pdf_buf.getvalue(), "body_adjusted.pdf", "application/pdf")
        st.download_button("🖼️ Скачать корпус (SVG)", svg_buf.getvalue(), "body_adjusted.svg", "image/svg+xml")

with tab2:
    st.subheader("Мерки для рукава")
    OG_s = st.number_input("ОГ (для рукава)", 70.0, 130.0, 103.0)
    pr_armhole_len_s = st.number_input("Длина проймы (по чертежу)", 40.0, 60.0, 48.0)
    sleeve_len = st.number_input("Длина рукава", 50.0, 70.0, 60.0)
    sleeve_width_bottom = st.number_input("Ширина низа рукава", 20.0, 35.0, 26.0)

    st.subheader("Те же особенности фигуры применяются к рукаву")

    if st.button("✂️ Построить рукав с учётом фигуры"):
        m_sleeve = Measurements(OG_s, 0, 0, 0, 0, 0, VPK, ShP, 0, 0, 0, pr_armhole_len_s, sleeve_len, sleeve_width_bottom)
        # для согласованности рукава с корпусом пересчитаем базовую глубину проймы
        body_raw = calc_base_grid(Measurements(OG_s, OT, OB, DTS, DTP, DI, VPK, ShP, Pruh, Ptal, Pbed, pr_armhole_len_s, sleeve_len, sleeve_width_bottom))
        body_adj = apply_figure_adjustments_body(body_raw, figure_options)
        sleeve_raw = calc_sleeve(m_sleeve)
        sleeve_adj = apply_figure_adjustments_sleeve(sleeve_raw, body_adj, figure_options)

        fig_sleeve = plot_sleeve(m_sleeve, sleeve_adj)
        st.pyplot(fig_sleeve)

        pdf_buf2, svg_buf2 = io.BytesIO(), io.BytesIO()
        fig_sleeve.savefig(pdf_buf2, format="pdf", bbox_inches="tight")
        fig_sleeve.savefig(svg_buf2, format="svg", bbox_inches="tight")
        st.download_button("📄 Скачать рукав (PDF)", pdf_buf2.getvalue(), "sleeve_adjusted.pdf", "application/pdf")
        st.download_button("🖼️ Скачать рукав (SVG)", svg_buf2.getvalue(), "sleeve_adjusted.svg", "image/svg+xml")
