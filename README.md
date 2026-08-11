# Atelier CAD (dress-pattern)

Параметрический конструктор выкроек для 3S Atelier.

**Контракт:** мерки + описание → `PatternSpec` (OpenRouter) → parametric drafting (ЕМКО/Мюллер) → PDF A4 / SVG / DXF.  
LLM **не** рисует крой.

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # при необходимости добавьте OPENROUTER_API_KEY
streamlit run app.py
```

Smoke без UI:

```bash
python scripts/smoke_pattern.py
```

## Документация для агента

- [`docs/ai/CONTEXT.md`](docs/ai/CONTEXT.md)
- [`docs/ai/BACKLOG.md`](docs/ai/BACKLOG.md)
- [`docs/ai/JOURNAL.md`](docs/ai/JOURNAL.md)
