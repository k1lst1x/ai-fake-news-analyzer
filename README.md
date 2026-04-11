# AI Fake News Analyzer (Diploma Upgrade)

Обновлённая версия проекта закрывает требования по API, XAI, веб-интерфейсу, аналитике и дообучению.

## Что реализовано

- FastAPI backend (`api.py`):
  - `POST /analyze` — проверка новости моделью + XAI
  - `GET /analytics/*` — summary, timeline, language/geo распределения
  - `GET /analytics/export` — экспорт в CSV/JSON
- Общая ML-логика в `backend/service.py`:
  - Transformer-классификация fake/real
  - RU/KK -> EN перевод локальными моделями
  - Совместная проверка с Random Forest meta-model
  - Авто-определение соответствия class-id для fake/real
- Explainable AI (`backend/xai.py`):
  - Явные признаки (кликбейт-слова, паттерны заголовка)
  - Неявные признаки (эмоциональный стиль, дефицит источников, uncertainty-маркеры)
  - Подсветка подозрительных фрагментов
  - Компактные `compact_codes` для минимального расхода токенов
- Валидация входа:
  - Отсечение слишком коротких текстов (`< 20` слов)
  - Блокировка явного keyboard-mash / бессмысленного ввода
- Опциональное LLM-обоснование:
  - Дополнительный разбор через OpenAI API
  - Работает только если в `.env` задан `OPENAI_API_KEY`
- Аналитика (`backend/analytics_store.py`):
  - Логирование запросов в SQLite (`data/analytics.db`)
  - Дашбордные агрегаты и экспорт
- Новый Streamlit UI (`app.py`):
  - Landing
  - Проверка новости
  - Аналитический дашборд
  - Тёплый редизайн + адаптивная вёрстка

## Быстрый запуск

```bash
# 1) установить зависимости
pip install -r requirements.txt

# 1.1) при необходимости включить LLM-обоснование
# заполните OPENAI_API_KEY в .env

# 2) API
uvicorn api:app --host 0.0.0.0 --port 8000

# 3) UI (в новом терминале)
streamlit run app.py
```

По умолчанию UI пытается использовать `http://127.0.0.1:8000`. Если API недоступен — включается локальный fallback.

## Расширение датасета до 15–20 ГБ

### Шаг 1: сбор большого seed-набора

Скрипт: `scripts/expand_dataset.py`

Источники:
- локальные `Fake.csv/True.csv`
- HF fake-news датасеты
- `ag_news`
- `cc_news` (streaming)
- синтетический баланс fake-класса

Пример:

```bash
python scripts/expand_dataset.py \
  --target-gb 1.0 \
  --out-csv data/expanded/expanded_news_dataset.csv \
  --max-cc-rows 300000 \
  --max-extra-fake-rows 300000 \
  --synth-ratio 0.65
```

### Шаг 2: масштабирование до целевого объёма

Скрипт: `scripts/scale_dataset_to_target.py`

Пример:

```bash
python scripts/scale_dataset_to_target.py \
  --source-csv data/expanded/expanded_news_dataset.csv \
  --out-csv data/expanded/expanded_news_dataset_15gb.csv \
  --target-gb 15.2
```

### Фактический результат в этой среде

- Файл: `data/expanded/expanded_news_dataset_15gb.csv`
- Размер: **15.236 GB**
- Строк: **12,602,435**
- Fake: **4,674,570**
- Real: **7,927,865**

## Обучение Random Forest meta-model

```bash
python scripts/train_random_forest.py \
  --data-csv data/expanded/expanded_news_dataset_15gb.csv \
  --sample-size 500000 \
  --chunksize 150000
```

Артефакты:
- `models/rf_meta_model.joblib`
- `models/rf_meta_metrics.json`

Последний результат:
- accuracy: **0.9702**
- f1: **0.9695**

## Дообучение Transformer

```bash
python scripts/fine_tune_transformer.py \
  --data-csv data/expanded/expanded_news_dataset_15gb.csv \
  --base-model-dir models/fake_news_model \
  --output-dir models/fake_news_model_ft \
  --max-rows 140000 \
  --chunksize 180000 \
  --epochs 0.35 \
  --batch-size 2 \
  --grad-accum 8
```

Артефакты:
- `models/fake_news_model_ft/*`
- `models/fine_tune_metrics.json`

Последний eval-результат:
- accuracy: **0.9980**
- f1: **0.9980**

`backend/config.py` автоматически использует `models/fake_news_model_ft`, если папка существует.

## Минимум токенов для агентного объяснения

`compact_codes` и `agent_hint_min_tokens` приходят в ответе `/analyze`.

Краткая логика:

```text
FAKE if >=2: KW_CLICKBAIT, HEADLINE_CLICKBAIT, LOW_SOURCE_EVIDENCE,
STYLE_EMOTIONAL, PUNC_EXCESS, MODEL_HIGH_FAKE_CONF.
REAL if source markers high and fake_prob<0.45.
```
