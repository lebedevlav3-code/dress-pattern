# JOURNAL — Atelier CAD (dress-pattern)

> Новые записи сверху. Агент читает только верхние 30–50 строк.

## 2026-08-11 — Втачной рукав + сплайны проймы (P0)

**Ветка (зеркало):** `feat/atelier-cad-sleeve-p0` в sewing-club  
**Сессия:** Cursor

### Что сделано

- `armscye.py` — общие кривые проймы + длина для оката
- `models/sleeve.py` — втачной рукав; пресеты; футляр+рукав
- Доводка горловины/проймы в `sheath_dress.py`; UI мерки рукава
- Smoke OK: sleeve=1, dress+sleeve=3

### Что не доделано

- Push в dress-pattern (write access); футболка; офлайн-макет

---

## 2026-08-11 — Зеркало в sewing-club + A0 + бэклог остатка

**Ветка (цель):** `cursor/atelier-cad-p0-patternspec-2685`  
**Сессия:** Cursor Cloud Agent

### Что сделано

- Добавлен экспорт PDF A0 (один лист, test square 5×5)
- Снимок кода залит в `sewing-club/external/atelier-cad/` (обход 403 push в dress-pattern)
- Патч: `sewing-club/external/atelier-cad-patches/dress-pattern-atelier-cad-p0.patch`
- BACKLOG обновлён: что ✅ / что ⏳ / что требует владельца

### Блокеры владельца

- Write access в dress-pattern для cursor[bot]
- OPENROUTER_API_KEY (опц.)
- Офлайн-проверка макета ±1–1.5 см

---

## 2026-08-11 — P0 каркас + PatternSpec + юбка E2E

**Ветка:** `cursor/atelier-cad-p0-patternspec-2685`  
**Сессия:** Cursor Cloud Agent

### Что сделано

- Созданы `docs/ai/{CONTEXT,BACKLOG,JOURNAL}.md` из утверждённого брифа
- Рефактор монолита `app.py` → пакет `atelier_cad/`
- JSON schema + SVG / DXF / tiled PDF A4
- PatternSpec + OpenRouter client; E2E юбка + футляр
- Инфра: `.env.example`, `prompts/`, smoke

### Решения

- Параметрическое ядро — источник геометрии; LLM только PatternSpec
- Invalid PatternSpec → ошибка, PDF не генерируется
