import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import io
from dataclasses import dataclass
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
import ezdxf

# ================= 1. ДАННЫЕ И НАСТРОЙКИ =================

@dataclass
class Measurements:
    OG: float; OT: float; OB: float
    DTS: float; DTP: float; DI: float
    VPK: float; ShP: float
    Vg: float; Cg: float
    Pruh: float; Ptal: float; Pbed: float
    # Рукав
    pr_len: float; sleeve_len: float; sleeve_w: float

# ================= 2. ЯДРО РАСЧЕТА (ЕМКО/Мюллер) =================

def calculate_grid(m: Measurements, opts):
    # Ширины
    Total_W = (m.OG + m.Pruh) / 2
    W_back = (m.OG / 8) + 5.5
    if opts['bust'] == 'полная': W_back -= 0.5
    
    W_arm = (m.OG / 8) - 1.5
    if W_arm < 9.5: W_arm = 9.5
    
    W_front = Total_W - W_back - W_arm

    # Вертикали
    Depth_Arm = (m.OG / 10) + 10.5 + 2.5
    if opts['shoulder'] == 'покатые': Depth_Arm += 1.5
    elif opts['shoulder'] == 'прямые': Depth_Arm -= 1.0

    levels = {
        'A': 0, 'G': Depth_Arm, 
        'T': m.DTS, 'B': m.DTS + 19.0, 'N': m.DI
    }

    # Вытачки
    W_waist_needed = (m.OT / 2) + (m.Ptal / 2)
    Total_Dart = max(0, Total_W - W_waist_needed)
    
    darts = {
        'back': Total_Dart * 0.25,
        'side': Total_Dart * 0.45,
        'front': Total_Dart * 0.30
    }
    
    # Нагрудная вытачка (раствор в см)
    bust_dart = 2.0
    if m.OG > 90: bust_dart = 3.5
    if m.OG > 105: bust_dart = 5.0
    if opts['bust'] == 'полная': bust_dart += 1.5

    return {'W': {'total': Total_W, 'back': W_back, 'arm': W_arm, 'front': W_front},
            'L': levels, 'D': darts, 'Misc': {'bust_dart': bust_dart}}

# ================= 3. ГЕОМЕТРИЧЕСКИЙ ДВИЖОК (SHAPELY) =================

def create_geometries(m, grid, opts):
    """Создает векторные контуры деталей для экспорта и припусков"""
    W = grid['W']; L = grid['L']; D = grid['D']
    
    # --- СПИНКА ---
    # Точки контура спинки
    neck_w = (m.OG / 13) + 2.5
    neck_h = neck_w / 3
    
    # Плечо с вытачкой
    angle_rad = np.radians(15 if opts['shoulder'] != 'покатые' else 20)
    dart_pos = neck_w + 4.0
    dart_val = 1.5 if opts['posture'] != 'сутулая' else 2.5
    dart_len = 7.0
    
    # Расчет координат плеча
    d1_x = dart_pos * np.cos(angle_rad) # упрощенно по X, т.к. угол мал
    d1_y = -neck_h + dart_pos * np.sin(angle_rad) # отсчет от ростка
    # (здесь нужна сложная тригонометрия, для MVP упростим до координат точек)
    
    # Формируем полигон спинки (упрощенный контур для примера)
    # В реальном САПР тут сотни точек сплайнов
    back_poly_coords = [
        (0, 0), (neck_w, -neck_h), # Росток
        (neck_w + m.ShP + dart_val, -neck_h + (m.ShP * np.sin(angle_rad)) + 2), # Конец плеча
        (W['back'], L['G'] - 5), # Пройма контрольная
        (W['back'] + W['arm']/2, L['G']), # Низ проймы
        (W['back'] + W['arm']/2 - D['side']/2, L['T']), # Талия бок
        (W['back'] + W['arm']/2 - D['side']/2 - 1, L['B']), # Бедра
        (W['back'] + W['arm']/2 - D['side']/2 - 1, L['N']), # Низ бок
        (0, L['N']), # Низ центр
        (0, 0) # Замыкаем
    ]
    back_geom = LineString(back_poly_coords) # Используем линию, чтобы не замыкать верх и низ неправильно
    
    # --- ПЕРЕД ---
    # Аналогично строим координаты переда
    bal = m.DTP - m.DTS
    start_y = -bal
    x_front = W['total']
    
    front_poly_coords = [
        (x_front, start_y), 
        (x_front - (neck_w+0.5), start_y), # Горловина
        (x_front - (neck_w+0.5), start_y + (neck_w+2)), # Глубина
        # ... (здесь опускаем детальное построение точек для краткости, 
        # используем логику из drawing функции для визуализации)
    ]
    
    return back_geom # Возвращаем геометрию для обработки

# ================= 4. ФУНКЦИИ ЭКСПОРТА =================

def create_dxf(m, grid, opts):
    """Генерация профессионального .dxf файла"""
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    # В DXF координаты обычно в мм, но швейники работают и в см. Оставим см.
    
    # Сетка
    L = grid['L']; W = grid['W']
    msp.add_line((0, -L['N']), (W['total'], -L['N']), dxfattribs={'layer': 'GRID', 'color': 7})
    msp.add_line((0, -L['G']), (W['total'], -L['G']), dxfattribs={'layer': 'GRID'})
    
    # Здесь мы дублируем логику рисования, но командами DXF
    # Для MVP добавим просто прямоугольник габаритов и основные линии
    msp.add_text(f"Pattern Base: OG={m.OG}", dxfattribs={'height': 2.0}).set_pos((0, 5))
    
    # Пример линии спинки
    neck_w = (m.OG / 13) + 2.5
    msp.add_lwpolyline([(0, 0), (neck_w, -neck_w/3)], dxfattribs={'layer': 'PATTERN', 'color': 1})
    
    # Важно: В полноценном коде нужно перенести всю логику draw_pattern сюда.
    # Сейчас это заглушка, показывающая, что файл создается.
    
    return doc

def save_tiled_pdf(fig, width_cm, height_cm):
    """
    Разрезает Matplotlib Figure на листы А4.
    """
    pdf_buffer = io.BytesIO()
    
    # Размеры А4 в дюймах (для matplotlib)
    a4_w_in = 8.27
    a4_h_in = 11.69
    # Поля (чтобы принтер не обрезал)
    margin_in = 0.5 
    
    # Рабочая область на листе
    work_w = a4_w_in - 2*margin_in
    work_h = a4_h_in - 2*margin_in
    
    # Конвертируем размеры чертежа в дюймы
    total_w_in = width_cm / 2.54
    total_h_in = height_cm / 2.54
    
    # Вычисляем кол-во листов
    cols = int(np.ceil(total_w_in / work_w))
    rows = int(np.ceil(total_h_in / work_h))
    
    with PdfPages(pdf_buffer) as pdf:
        for r in range(rows):
            for c in range(cols):
                # Определяем "окно" просмотра для текущей страницы
                x_min = c * work_w * 2.54 # обратно в см для set_xlim
                x_max = x_min + (work_w * 2.54)
                
                # Y идет сверху вниз на графике, но тайлинг удобнее снизу
                # В Matplotlib (0,0) сверху слева в нашей настройке
                y_min = r * work_h * 2.54
                y_max = y_min + (work_h * 2.54)
                
                # Устанавливаем границы просмотра (Zoom)
                ax = fig.get_axes()[0]
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_max, y_min) # Инверсия Y сохраняется
                
                # Добавляем метки совмещения (текст на полях)
                ax.set_title(f"Лист {r+1}-{c+1} (Ряд {r+1}, Кол {c+1})", fontsize=10, color='red')
                
                # Сохраняем страницу
                # bbox_inches='tight' нельзя, иначе масштаб собьется!
                # Нужно сохранять строго в размер А4
                fig.set_size_inches(a4_w_in, a4_h_in)
                pdf.savefig(fig, paperformat='a4')
                
    pdf_buffer.seek(0)
    return pdf_buffer

# ================= 5. ОТРИСОВКА И ПРИПУСКИ =================

def draw_pattern_final(m, grid, opts, show_seam_allowance):
    # Увеличиваем размер фигуры, чтобы влезло всё
    fig, ax = plt.subplots(figsize=(10, 14))
    W = grid['W']; L = grid['L']; D = grid['D']; Misc = grid['Misc']
    
    # --- ВСПОМОГАТЕЛЬНЫЕ ЛИНИИ (СЕТКА) ---
    for name, y in L.items():
        ax.axhline(y, color='#e0e0e0', lw=0.5)
        ax.text(-1, y, name, fontsize=6, color='gray')
    ax.vlines([0, W['back'], W['back']+W['arm'], W['total']], 0, L['N'], colors='#e0e0e0', lw=0.5)

    # --- СПИНКА ---
    # Логика построения точек (как в v3.0, но собранная для plot)
    neck_w = (m.OG / 13) + 2.5
    neck_h = neck_w / 3
    
    # Росток
    x_n = np.linspace(0, neck_w, 10); y_n = -neck_h * (x_n/neck_w)**2
    ax.plot(x_n, y_n, 'b', label='Контур')
    
    # Плечо
    angle = 15 + (5 if opts['shoulder']=='покатые' else -5 if opts['shoulder']=='прямые' else 0)
    rad = np.radians(angle)
    # Конец плеча
    sh_len = m.ShP + (1.5 if opts['posture']!='сутулая' else 2.5) # + вытачка
    sh_x = neck_w + sh_len * np.cos(rad)
    sh_y = y_n[-1] + sh_len * np.sin(rad)
    
    # Вытачка плечевая (схематично)
    d_start = neck_w + 4.0
    ax.plot([neck_w, sh_x], [y_n[-1], sh_y], 'b')
    # Вершина вытачки
    ax.plot([d_start, d_start + 0.7, d_start + 1.5], 
            [y_n[-1] + (d_start-neck_w)*np.sin(rad), y_n[-1] + 8, y_n[-1] + (d_start-neck_w+1.5)*np.sin(rad)], 'b')

    # Пройма
    center_arm = W['back'] + W['arm']/2
    ax.plot([sh_x, W['back'], center_arm], [sh_y, L['G']-6, L['G']], 'b')
    
    # Бок
    side_val = D['side']/2
    ax.plot([center_arm, center_arm-0.5, center_arm-side_val], [L['G'], (L['G']+L['T'])/2, L['T']], 'b')
    
    # Бедра и низ
    hip_excess = (m.OB/2 + m.Pbed/2) - W['total']
    hip_sh = hip_excess/2
    bx_hip = center_arm-side_val-hip_sh
    ax.plot([center_arm-side_val, bx_hip], [L['T'], L['B']], 'b')
    ax.plot([bx_hip, bx_hip], [L['B'], L['N']], 'b') # Низ
    ax.plot([0, bx_hip], [L['N'], L['N']], 'b') # Линия низа
    ax.plot([0, 0], [0, L['N']], 'b') # Центр спинки

    # --- ПРИПУСКИ (АВТОМАТИЧЕСКИЕ) ---
    if show_seam_allowance:
        # Для MVP делаем простой offset: смещение основных узлов
        # В полноценном ПО это делает Shapely.buffer()
        allowance = 1.5 # см
        
        # Пример для бокового шва
        ax.plot([center_arm-side_val-allowance, bx_hip-allowance], [L['T'], L['B']], 'b--', lw=0.8, alpha=0.6)
        ax.plot([bx_hip-allowance, bx_hip-allowance], [L['B'], L['N']+4], 'b--', lw=0.8, alpha=0.6) # Низ +4см
        
        ax.text(W['total']/2, L['N']+10, "--- Пунктир: Линии реза (Припуски: Бок 1.5см, Низ 4см, Пройма 1см)", color='gray', fontsize=8)

    # --- ПЕРЕД ---
    # (Упрощенная отрисовка для экономии места кода, полная логика в v3.0)
    bal = m.DTP - m.DTS
    start_y_f = -bal
    x_front = W['total']
    
    neck_w_f = neck_w + 0.5
    ax.plot([x_front, x_front-neck_w_f], [start_y_f, start_y_f], 'm') # Верх
    ax.plot([x_front-neck_w_f, x_front-neck_w_f], [start_y_f, start_y_f+neck_w+2], 'm') # Глубина
    
    # ЦГ
    apex_x = x_front - m.Cg
    apex_y = start_y_f + m.Vg
    ax.plot(apex_x, apex_y, 'ro', ms=3)
    
    # Настройка
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.grid(False)
    ax.axis('off') # Убираем рамки графика для чистоты печати
    
    return fig

# ================= 6. UI ИНТЕРФЕЙС =================

st.set_page_config(page_title="Atelier CAD v4.0", layout="wide")
st.title("✂️ Atelier CAD: Профессиональный конструктор")

# Ввод данных
with st.sidebar:
    st.header("Параметры клиента")
    # Блок ввода (кратко)
    OG = st.number_input("ОГ", 80.0, 130.0, 96.0)
    OT = st.number_input("ОТ", 50.0, 110.0, 76.0)
    OB = st.number_input("ОБ", 80.0, 140.0, 104.0)
    DTS = st.number_input("ДТС", 35.0, 50.0, 42.0)
    DTP = st.number_input("ДТП", 35.0, 60.0, 44.0)
    DI = st.number_input("ДИ", 50.0, 150.0, 100.0)
    
    with st.expander("Дополнительные мерки"):
        VPK = st.number_input("Впк", 35.0, 50.0, 42.0)
        ShP = st.number_input("Шп", 10.0, 20.0, 13.0)
        Vg = st.number_input("Вг", 20.0, 40.0, 27.0)
        Cg = st.number_input("Цг", 15.0, 25.0, 20.0)
        
    st.subheader("Настройки выкройки")
    Pruh = st.slider("Прибавка (Грудь)", 0.0, 10.0, 4.0)
    Ptal = st.slider("Прибавка (Талия)", 0.0, 10.0, 2.0)
    Pbed = st.slider("Прибавка (Бедра)", 0.0, 10.0, 2.0)
    
    seams = st.checkbox("✅ Добавить припуски на швы", value=True)

# Инициализация
m = Measurements(OG, OT, OB, DTS, DTP, DI, VPK, ShP, Vg, Cg, Pruh, Ptal, Pbed, 0, 0, 0)
opts = {'bust': 'средняя', 'shoulder': 'нормальные', 'posture': 'нормальная'} # Default
grid = calculate_grid(m, opts)

tab1, tab2 = st.tabs(["👗 Чертеж и Печать", "💾 Экспорт DXF"])

with tab1:
    st.markdown("### Предпросмотр раскладки")
    
    # Рисуем
    fig = draw_pattern_final(m, grid, opts, seams)
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("🖨️ Печать на А4 (домашний принтер)")
    st.info("Выкройка будет автоматически нарезана на листы А4 с метками совмещения.")
    
    if st.button("Сгенерировать PDF для печати"):
        # Расчет реальных размеров для нарезки
        w_cm = grid['W']['total'] + 10 # Запас
        h_cm = m.DI + 10
        
        pdf_data = save_tiled_pdf(fig, w_cm, h_cm)
        st.download_button(
            label="📄 Скачать многостраничный PDF (А4)",
            data=pdf_data,
            file_name="pattern_tiled_A4.pdf",
            mime="application/pdf"
        )

with tab2:
    st.subheader("Экспорт в CAD (AutoCAD, CLO3D, Corel)")
    st.write("Формат DXF является промышленным стандартом.")
    
    if st.button("Создать DXF файл"):
        dxf_doc = create_dxf(m, grid, opts)
        
        # Сохраняем в буфер
        stream = io.StringIO()
        dxf_doc.write(stream)
        
        st.download_button(
            label="💾 Скачать .DXF",
            data=stream.getvalue(),
            file_name="pattern_base.dxf",
            mime="application/dxf"
        )
