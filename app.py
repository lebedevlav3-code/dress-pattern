"""Atelier CAD — Streamlit UI (тонкая оболочка над atelier_cad)."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from atelier_cad.export import export_all, render_preview_figure
from atelier_cad.measurements import DEFAULT_MEASUREMENTS, FigureOptions, Measurements
from atelier_cad.models import build_from_spec
from atelier_cad.openrouter import describe_to_spec_or_preset, is_configured
from atelier_cad.pattern_spec import (
    PatternSpecError,
    parse_pattern_spec,
    preset_dress_sheath_boat_midi,
    preset_dress_sheath_with_sleeve,
    preset_skirt_a_line,
    preset_skirt_straight,
    preset_sleeve_set_in_long,
)

load_dotenv()

st.set_page_config(page_title="Atelier CAD", layout="wide")
st.title("Atelier CAD — параметрический конструктор")
st.caption(
    "Ядро: мерки → PatternSpec → parametric drafting → PDF/SVG/DXF. "
    "AI не рисует крой, только предлагает параметры. "
    "Посадка без макета не обещается."
)

with st.expander("Для кого и когда использовать", expanded=False):
    st.markdown(
        """
**Для кого:** ателье / MTM / владелец — индивидуальный крой под клиента, PDF и DXF.

**Когда:** нужен параметрический чертёж по методике (ЕМКО/Мюллер), не раздача ученицам на сайте клуба.

**Когда лучше Pattern Studio клуба:** урок МК, у учениц разные фигуры, модель уже в каталоге FreeSewing
(или идёт через Brief → новый design). Studio — LMS «мерки в ЛК → PDF к уроку»; CAD туда не встраиваем.

Два продукта, не сливать (мосты только мерки · Brief · CTA).
"""
    )

# ---------- Sidebar: measurements & figure ----------
with st.sidebar:
    st.header("Мерки клиента (см)")
    if "meas_preset" not in st.session_state:
        st.session_state.meas_preset = DEFAULT_MEASUREMENTS.to_dict()

    if st.button("Сбросить мерки к пресету"):
        st.session_state.meas_preset = DEFAULT_MEASUREMENTS.to_dict()

    mp = st.session_state.meas_preset
    OG = st.number_input("ОГ", 70.0, 150.0, float(mp["OG"]))
    OT = st.number_input("ОТ", 50.0, 140.0, float(mp["OT"]))
    OB = st.number_input("ОБ", 70.0, 160.0, float(mp["OB"]))
    DTS = st.number_input("ДТС", 30.0, 55.0, float(mp["DTS"]))
    DTP = st.number_input("ДТП", 30.0, 65.0, float(mp["DTP"]))
    DI = st.number_input("ДИ / длина изделия", 20.0, 180.0, float(mp["DI"]))

    with st.expander("Дополнительные мерки"):
        VPK = st.number_input("Впк", 30.0, 55.0, float(mp["VPK"]))
        ShP = st.number_input("Шп", 8.0, 22.0, float(mp["ShP"]))
        Vg = st.number_input("Вг", 18.0, 45.0, float(mp["Vg"]))
        Cg = st.number_input("Цг", 12.0, 28.0, float(mp["Cg"]))
        sleeve_len = st.number_input(
            "Длина рукава (0 = по пресету стиля)",
            0.0,
            80.0,
            float(mp.get("sleeve_len", 0.0)),
        )
        sleeve_w = st.number_input(
            "Окружность руки с прибавкой (0 = от проймы)",
            0.0,
            60.0,
            float(mp.get("sleeve_w", 0.0)),
        )

    st.subheader("Прибавки на свободу")
    Pruh = st.slider("Грудь", 0.0, 12.0, float(mp["Pruh"]))
    Ptal = st.slider("Талия", 0.0, 12.0, float(mp["Ptal"]))
    Pbed = st.slider("Бёдра", 0.0, 12.0, float(mp["Pbed"]))

    st.subheader("Опции фигуры")
    bust = st.selectbox("Бюст", ["маленькая", "средняя", "полная"], index=1)
    shoulder = st.selectbox("Плечи", ["покатые", "нормальные", "прямые"], index=1)
    posture = st.selectbox("Осанка", ["сутулая", "нормальная", "перегибистая"], index=1)

    st.session_state.meas_preset = {
        "OG": OG,
        "OT": OT,
        "OB": OB,
        "DTS": DTS,
        "DTP": DTP,
        "DI": DI,
        "VPK": VPK,
        "ShP": ShP,
        "Vg": Vg,
        "Cg": Cg,
        "Pruh": Pruh,
        "Ptal": Ptal,
        "Pbed": Pbed,
        "sleeve_len": sleeve_len,
        "sleeve_w": sleeve_w,
    }

m = Measurements(
    OG=OG,
    OT=OT,
    OB=OB,
    DTS=DTS,
    DTP=DTP,
    DI=DI,
    VPK=VPK,
    ShP=ShP,
    Vg=Vg,
    Cg=Cg,
    Pruh=Pruh,
    Ptal=Ptal,
    Pbed=Pbed,
    sleeve_len=sleeve_len,
    sleeve_w=sleeve_w,
)
figure = FigureOptions(bust=bust, shoulder=shoulder, posture=posture)

# ---------- PatternSpec source ----------
tab_model, tab_ai, tab_checklist = st.tabs(
    ["Модель и чертёж", "AI → PatternSpec", "Перед кроем"]
)

with tab_ai:
    st.markdown("### Текст → PatternSpec (OpenRouter)")
    if is_configured():
        st.success("OPENROUTER_API_KEY найден — будет вызван OpenRouter.")
    else:
        st.warning(
            "Ключ OPENROUTER_API_KEY не задан — сработает офлайн-эвристика / пресет. "
            "Секреты только в `.env`, не в коде."
        )
    description = st.text_area(
        "Описание изделия",
        value="платье-футляр миди, лодочка",
        height=100,
        help="Без мерок и ФИО. Пример: «юбка-трапеция миди»",
    )
    if st.button("Получить PatternSpec"):
        try:
            spec = describe_to_spec_or_preset(description)
            st.session_state.pattern_spec = spec.model_dump(mode="json")
            st.success("PatternSpec получен")
        except PatternSpecError as exc:
            st.error(str(exc))
            st.stop()

with tab_model:
    st.markdown("### Выбор модели (пресет) или текущий PatternSpec")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Юбка прямая"):
            st.session_state.pattern_spec = preset_skirt_straight().model_dump(mode="json")
    with c2:
        if st.button("Юбка-трапеция"):
            st.session_state.pattern_spec = preset_skirt_a_line().model_dump(mode="json")
    with c3:
        if st.button("Футляр миди, лодочка"):
            st.session_state.pattern_spec = preset_dress_sheath_boat_midi().model_dump(mode="json")
    c4, c5, _ = st.columns(3)
    with c4:
        if st.button("Рукав втачной"):
            st.session_state.pattern_spec = preset_sleeve_set_in_long().model_dump(mode="json")
    with c5:
        if st.button("Футляр + рукав"):
            st.session_state.pattern_spec = preset_dress_sheath_with_sleeve().model_dump(
                mode="json"
            )

    if "pattern_spec" not in st.session_state:
        st.session_state.pattern_spec = preset_skirt_straight().model_dump(mode="json")

    st.json(st.session_state.pattern_spec)

    build = st.button("Построить выкройку", type="primary")
    if build or st.session_state.get("auto_built"):
        try:
            pieces = build_from_spec(m, st.session_state.pattern_spec, figure)
            st.session_state.pieces_ready = True
            st.session_state.last_export = export_all(
                pieces,
                measurements=m,
                figure=figure,
                spec=parse_pattern_spec(st.session_state.pattern_spec),
                title="Atelier CAD",
            )
            st.session_state.preview_fig = render_preview_figure(pieces)
            st.session_state.auto_built = True
        except PatternSpecError as exc:
            st.session_state.pieces_ready = False
            st.error(f"PatternSpec отклонён — PDF не создан. {exc}")
        except Exception as exc:  # noqa: BLE001
            st.session_state.pieces_ready = False
            st.error(f"Ошибка построения: {exc}")

    if st.session_state.get("pieces_ready") and st.session_state.get("preview_fig") is not None:
        st.pyplot(st.session_state.preview_fig)
        exp = st.session_state.last_export
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.download_button(
            "PDF A4 (tiled)",
            data=exp["pdf"],
            file_name="pattern_A4.pdf",
            mime="application/pdf",
        )
        c2.download_button(
            "PDF A0",
            data=exp["pdf_a0"],
            file_name="pattern_A0.pdf",
            mime="application/pdf",
        )
        c3.download_button(
            "SVG",
            data=exp["svg"],
            file_name="pattern.svg",
            mime="image/svg+xml",
        )
        c4.download_button(
            "DXF",
            data=exp["dxf"],
            file_name="pattern.dxf",
            mime="application/dxf",
        )
        c5.download_button(
            "JSON schema",
            data=exp["json"],
            file_name="pattern.json",
            mime="application/json",
        )

with tab_checklist:
    st.markdown("### Чеклист перед кроем")
    st.checkbox("Мерки свежие (не старше 3 месяцев) и сняты по одной методике")
    st.checkbox("Распечатан test square 5×5 см — сторона ровно 5 см")
    st.checkbox("Листы склеены по overlap-меткам без перекоса")
    st.checkbox("Контрольные мерки на бумаге сверены с паспортом (±1–1.5 см)")
    st.checkbox("Макет из макетной ткани до кроя основной")
    st.info(
        "Посадка проверяется на макете. PDF не обещает идеальную посадку с первого раза."
    )
    st.markdown(
        "Нужна помощь на примерке? Заявка Fitting Concierge — в ЛК (P2), "
        "пока можно написать куратору пространства 3S Atelier."
    )
