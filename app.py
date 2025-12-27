import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io
from dataclasses import dataclass

# ================= ДАННЫЕ =================

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
    # Для рукава
    pr_len: float
    sleeve_len: float
    sleeve_w: float

# ================= РАСЧЕТЫ =================

def calculate_grid(m: Measurements, opts):
    """Рассчитывает координаты основных линий и точек"""
    
    # 1. Ширины участков (формулы ЕМКО упрощенные)
    Sg = m.OG / 2  # Полуобхват
    Pg = m.Pruh    # Прибавка
    
    # Распределение ширины сетки: Спина ~19%, Пройма ~12.5%, Перед - остаток
    # (Используем коэффициенты для адаптации под размеры)
    total_width = (m.OG + m.Pruh) / 2
    
    # Базовые ширины
    w_back = (m.OG / 8) + 5.5
    w_arm = (m.OG / 8) - 1.5
    if w_arm < 9: w_arm = 9 # защита для малых размеров
    w_front = total_width - w_back - w_arm
    
    # Корректировки из "Особенностей фигуры"
    if opts['bust'] == 'полная':
        w_front += 1.0
        w_back -= 0.5
    if opts['bust'] == 'малая':
        w_front -= 0.5
        
    # Глубина проймы (расчетная)
    depth_arm = (m.OG / 10) + 10.5 + 2.0 # База + свобода
    if opts['shoulder'] == 'покатые': depth_arm += 1.0
    if opts['shoulder'] == 'прямые': depth_arm -= 0.5

    # Уровни по вертикали (Y=0 - это 7-й шейный позвонок)
    levels = {
        'A': 0,                     # Шея
        'G': depth_arm,             # Грудь (глубина проймы)
        'T': m.DTS,                 # Талия
        'B': m.DTS + 18.0,          # Бедра (стандарт 18-20 см от талии)
        'N': m.DI                   # Низ
    }

    # Расчет талиевых вытачек (суммарный раствор)
    # Ширина сетки - (Полуобхват талии + Прибавка)
    w_waist_grid = total_width
    w_waist_needed = (m.OT / 2) + (m.Ptal / 2)
    total_dart = w_waist_grid - w_waist_needed
    
    # Распределение вытачек: 50% бок, 30% спинка, 20% перед
    darts = {
        'side': total_dart * 0.5,
        'back': total_dart * 0.3,
        'front': total_dart * 0.2
    }
    
    return {
        'widths': {'total': total_width, 'back': w_back, 'arm': w_arm, 'front': w_front},
        'levels': levels,
        'darts': darts
    }

# ================= ОТРИСОВКА =================

def draw_pattern(m: Measurements, grid, opts):
    fig, ax = plt.subplots(figsize=(8, 12))
    
    W = grid['widths']
    L = grid['levels']
    D = grid['darts']
    
    # Границы зон по X
    x_back_edge = W['back']
    x_front_edge = W['back'] + W['arm']
    x_total = W['total']
    
    # --- 1. СЕТКА ---
    # Горизонтали
    for label, y in L.items():
        ax.axhline(y, color='lightgray', linestyle='--', linewidth=0.8)
        ax.text(-1, y, label, va='center', ha='right', fontsize=8, color='gray')
        
    # Вертикали зон
    ax.vlines([0, x_back_edge, x_front_edge, x_total], 0, L['N'], colors='lightgray', linestyles='--')

    # --- 2. СПИНКА (Синий) ---
    # Горловина
    neck_w = (m.OG / 13) + 2.5
    neck_h = neck_w / 3
    x_n = np.linspace(0, neck_w, 20)
    y_n = -neck_h * (x_n/neck_w)**2 # парабола ростка
    ax.plot(x_n, y_n, 'b')
    
    # Плечо (с учетом Впк)
    # Упрощенная геометрия: используем Впк как проверку высоты конца плеча
    # Примерно: конец плеча = точка пересечения дуги R=Шп от шеи и R=Впк от центра талии
    # Для графика упростим: найдем точку через косинус угла наклона
    
    # Стандартный наклон ~15-20 градусов, корректируем по фигуре
    angle_deg = 15
    if opts['shoulder'] == 'покатые': angle_deg += 5
    if opts['shoulder'] == 'прямые': angle_deg -= 5
    
    angle_rad = np.radians(angle_deg)
    sh_x = neck_w + m.ShP * np.cos(angle_rad)
    sh_y = y_n[-1] + m.ShP * np.sin(angle_rad)
    
    # Проверка на Впк (визуализация контроля)
    # Дистанция от (sh_x, sh_y) до (0, L['T']) должна быть ~ Впк
    # Здесь мы просто рисуем линию плеча
    ax.plot([neck_w, sh_x], [y_n[-1], sh_y], 'b', lw=1.5)
    
    # Пройма спинки
    ax.plot([sh_x, x_back_edge, x_back_edge + W['arm']/2], 
            [sh_y, L['G'] - 5, L['G']], 'b') # Схематично

    # --- 3. ПЕРЕД (Малиновый) ---
    # Баланс (ДТП - ДТС) определяет, насколько выше/ниже точка основания шеи
    balance = m.DTP - m.DTS
    start_y = -balance # Если ДТП > ДТС, точка уходит вверх (в минус по Y)
    
    # Горловина переда
    neck_w_f = neck_w + 0.5
    neck_h_f = neck_w_f + 1.5
    
    # Центр переда справа (x_total)
    x_nf = np.linspace(x_total - neck_w_f, x_total, 20)
    y_nf = start_y + neck_h_f * (1 - ((x_nf - (x_total - neck_w_f))/neck_w_f)**2)**0.5 # Окружность
    ax.plot(x_nf, y_nf, 'm')
    
    # Плечо переда
    # Наклон переда обычно больше (около 25 град)
    sh_f_drop = 4.0 # см вниз от высшей точки
    if opts['shoulder'] == 'покатые': sh_f_drop += 1.0
    
    sh_fx_start = x_total - neck_w_f
    sh_fy_start = y_nf[0] # Высшая точка горловины
    
    # Конец плеча (упрощенно по X)
    sh_fx_end = sh_fx_start - (m.ShP * 0.95) # проекция
    sh_fy_end = sh_fy_start + sh_f_drop
    
    ax.plot([sh_fx_start, sh_fx_end], [sh_fy_start, sh_fy_end], 'm', lw=1.5)
    
    # Пройма переда
    ax.plot([sh_fx_end, x_front_edge, x_back_edge + W['arm']/2],
            [sh_fy_end, L['G'] - 6, L['G']], 'm')

    # --- 4. БОКОВЫЕ ШВЫ И ТАЛИЯ ---
    # Середина проймы
    mid_arm = x_back_edge + W['arm']/2
    
    # Расчет заужения бока (половина суммарной вытачки делится на 2 бока)
    side_indent = D['side'] / 2
    
    # Линия бока Спинки (Синяя)
    ax.plot([mid_arm, mid_arm - 1, mid_arm - side_indent], 
            [L['G'], L['G'] + (L['T']-L['G'])/2, L['T']], 'b') # до талии
            
    # Линия бока Переда (Малиновая)
    ax.plot([mid_arm, mid_arm + 1, mid_arm + side_indent], 
            [L['G'], L['G'] + (L['T']-L['G'])/2, L['T']], 'm') # до талии
            
    # Бедра (расширение)
    # Расчет излишка/недостатка по бедрам
    w_hips_grid = W['total']
    w_hips_needed = (m.OB / 2) + (m.Pbed / 2)
    hips_diff = w_hips_needed - w_hips_grid
    
    hips_indent = hips_diff / 2
    
    # От талии до бедер
    ax.plot([mid_arm - side_indent, mid_arm - side_indent - hips_indent], [L['T'], L['B']], 'b')
    ax.plot([mid_arm + side_indent, mid_arm + side_indent + hips_indent], [L['T'], L['B']], 'm')
    
    # Низ (вертикально вниз от бедер)
    hip_end_b = mid_arm - side_indent - hips_indent
    hip_end_f = mid_arm + side_indent + hips_indent
    ax.plot([hip_end_b, hip_end_b], [L['B'], L['N']], 'b')
    ax.plot([hip_end_f, hip_end_f], [L['B'], L['N']], 'm')

    # Настройка вида
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title("Чертеж основы (схема)")
    return fig

def draw_sleeve(m: Measurements, body_depth, opts):
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Высота оката (зависит от длины проймы и корректировок)
    # Чем глубже пройма, тем выше должен быть окат для компенсации
    H_okat = m.pr_len / 2.5 # Примерная формула (ЛП / 2.5 ~ 14-16 см)
    
    # Корректировка оката от глубины проймы корпуса
    base_depth = 23.0
    diff = body_depth - base_depth
    H_okat += diff * 0.5 
    
    W_sleeve = (m.OG / 3) + 2 # Ширина рукава вверху
    if opts['bust'] == 'полная': W_sleeve += 1.5
    
    # Построение
    w_half = W_sleeve / 2
    
    # Сетка
    ax.plot([-w_half, w_half], [0, 0], 'k--', lw=0.5) # Линия высоты оката
    ax.plot([0, 0], [-H_okat, m.sleeve_len - H_okat], 'k-.', lw=0.5) # Центр
    
    # Окат (Волна)
    # Левая часть (спинка) - более пологая
    x_back = np.linspace(-w_half, 0, 20)
    # Параметрическая кривая (упрощенно)
    y_back = -H_okat * np.sin(np.pi * (x_back + w_half) / w_half / 2)
    ax.plot(x_back, y_back, 'b', label='К спинке')
    
    # Правая часть (перед) - более крутая выемка
    x_front = np.linspace(0, w_half, 20)
    y_front = -H_okat * np.sin(np.pi * (w_half - x_front) / w_half / 2)
    # Чуть углубляем переднюю часть вручную (искажение синусоиды)
    y_front = y_front * (1 + 0.2 * np.sin(np.pi * x_front / w_half)) 
    
    ax.plot(x_front, y_front, 'm', label='К переду')
    
    # Боковые швы
    bottom_w_half = m.sleeve_w / 2
    h_total = m.sleeve_len - H_okat
    
    ax.plot([-w_half, -bottom_w_half], [0, h_total], 'b')
    ax.plot([w_half, bottom_w_half], [0, h_total], 'm')
    ax.plot([-bottom_w_half, bottom_w_half], [h_total, h_total], 'k')

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Чертеж рукава")
    
    return fig

# ================= ИНТЕРФЕЙС =================

st.set_page_config(page_title="Конструктор лекал", layout="wide")
st.title("✂️ Интерактивный конструктор основы платья")

# Сайдбар с настройками
with st.sidebar:
    st.header("1. Мерки (см)")
    OG = st.number_input("Обхват Груди (ОГ)", 80.0, 140.0, 96.0)
    OT = st.number_input("Обхват Талии (ОТ)", 50.0, 120.0, 76.0)
    OB = st.number_input("Обхват Бедер (ОБ)", 80.0, 140.0, 104.0)
    DTS = st.number_input("Длина Спины (ДТС)", 35.0, 50.0, 42.0)
    DTP = st.number_input("Длина Переда (ДТП)", 35.0, 60.0, 44.0)
    DI = st.number_input("Длина Изделия (ДИ)", 50.0, 150.0, 100.0)
    
    st.markdown("---")
    VPK = st.number_input("Высота плеча косая (Впк)", 30.0, 50.0, 42.0)
    ShP = st.number_input("Ширина плеча (Шп)", 10.0, 20.0, 13.0)
    
    st.header("2. Прибавки")
    Pruh = st.slider("К полуобхвату груди", 0.0, 10.0, 4.0)
    Ptal = st.slider("К полуобхвату талии", 0.0, 10.0, 2.0)
    Pbed = st.slider("К полуобхвату бедер", 0.0, 10.0, 2.0)
    
    st.header("3. Особенности")
    opt_sh = st.selectbox("Плечи", ["нормальные", "покатые", "прямые"])
    opt_posture = st.selectbox("Осанка", ["нормальная", "сутулая", "перегибистая"])
    opt_bust = st.selectbox("Грудь", ["средняя", "малая", "полная"])
    
    opts = {'shoulder': opt_sh, 'posture': opt_posture, 'bust': opt_bust}

# Основной экран
tab1, tab2 = st.tabs(["👗 Корпус", "👕 Рукав"])

# Создаем объект мерок (заглушки для рукава пока 0)
m = Measurements(OG, OT, OB, DTS, DTP, DI, VPK, ShP, Pruh, Ptal, Pbed, 0, 0, 0)
grid = calculate_grid(m, opts)

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        fig_body = draw_pattern(m, grid, opts)
        st.pyplot(fig_body)
    with col2:
        st.info("💡 **Пояснение:**\nСиняя линия — контур спинки.\nМалиновая — контур переда.\nПунктиры — конструктивная сетка.")
        st.write(f"**Расчеты:**")
        st.write(f"Ширина спинки: {grid['widths']['back']:.1f} см")
        st.write(f"Ширина проймы: {grid['widths']['arm']:.1f} см")
        st.write(f"Ширина переда: {grid['widths']['front']:.1f} см")
        st.write(f"Раствор боковой вытачки: {grid['darts']['side']:.1f} см")
        
        # Экспорт
        fn = "pattern_body"
        img = io.BytesIO()
        fig_body.savefig(img, format='pdf')
        st.download_button("Скачать PDF", img.getvalue(), f"{fn}.pdf", "application/pdf")

with tab2:
    st.subheader("Параметры рукава")
    c1, c2, c3 = st.columns(3)
    with c1: pr_len = st.number_input("Длина проймы (измерьте по чертежу)", 30.0, 60.0, 45.0)
    with c2: sl_len = st.number_input("Длина рукава", 40.0, 70.0, 60.0)
    with c3: sl_w = st.number_input("Ширина низа", 20.0, 40.0, 24.0)
    
    m_sleeve = Measurements(OG, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, pr_len, sl_len, sl_w)
    fig_sleeve = draw_sleeve(m_sleeve, grid['levels']['G'], opts) # Передаем глубину проймы корпуса
    
    st.pyplot(fig_sleeve)
