# Процес аналізу датасету (v6)

Опис того, як `analyze.py` незалежно аналізує згенерований датасет діалогів.

## Загальна ідея

Аналізатор бере готовий датасет (`data/dataset.json`) і проганяє кожен діалог через LLM,
щоб отримати незалежну оцінку якості. Результати не копіюються з метаданих генератора —
LLM дивиться на діалог "з нуля". Це дозволяє порівняти, наскільки точно
аналізатор розпізнає інтенти, задоволеність і помилки агента.

## Архітектура: Two-Tier Batch Analysis

Аналізатор використовує **двоярусну** архітектуру для оптимізації швидкості та точності:

### Tier 1: Intent + Satisfaction (batch_size=15)
- Класифікує намір клієнта та рівень задоволеності
- Більший розмір батчу (15 діалогів), бо виходу менше (2 поля)
- Chain-of-thought: модель спочатку пояснює, що питає клієнт, потім класифікує

### Tier 2: Quality + Mistakes + HD (batch_size=10)
- Оцінює якість агента, знаходить помилки, виявляє приховане незадоволення
- Менший розмір батчу (10 діалогів), бо виходу більше (3 поля, списки помилок)
- Chain-of-thought: модель оцінює, чи була проблема вирішена і наскільки добре

### Чому два яруси, а не один?
- **Швидкість**: менше токенів на виході для кожного LLM-виклику → швидше генерація
- **Точність**: кожен ярус фокусується на своїй задачі → менше "розмивання" уваги моделі
- **Надійність**: якщо один ярус не повернув результат, інший все одно працює

## Кроки

### 1. Завантаження датасету

Читається файл датасету (за замовчуванням `data/dataset.json`, можна вказати інший через аргумент командного рядка).

### 2. Препроцесинг діалогів

`_preprocess_dialog()` замінює шаблонні змінні (наприклад, `{{Order Number}}`, `{{Customer Name}}`)
на реалістичні значення з таблиці `TEMPLATE_REPLACEMENTS` у `constants.py`.
Це покращує розуміння контексту моделлю.

Також обрізає відповіді агента до 300 символів (`MAX_AGENT_CHARS`) щоб зменшити розмір промпту.

### 3. Аналіз через LLM

`DatasetAnalyzer.analyze_batch()` обробляє діалоги в два яруси:

**Tier 1** — `_run_tier1_batch()`:
- Формує промпт з описами інтентів, прикладами, правилами disambiguation
- LLM повертає JSON з `intent` та `satisfaction` для кожного діалогу
- Валідація: `_validate_tier1()` перевіряє intent проти `VALID_INTENTS`, satisfaction проти допустимих значень

**Tier 2** — `_run_tier2_batch()`:
- Формує промпт з рубрикою якості, описами помилок, правилами HD
- LLM повертає JSON з `quality_score`, `agent_mistakes`, `hidden_dissatisfaction`
- Валідація: `_validate_tier2()` перевіряє score в діапазоні 1-5, фільтрує помилки

**Мерж** — результати обох ярусів об'єднуються. Якщо для діалогу відсутній один ярус, використовується fallback (single-dialog prompt).

### 4. Визначення 5 метрик

Для кожного діалогу LLM визначає 5 метрик:

| Метрика | Значення | Опис |
|---------|----------|------|
| `intent` | одне з `VALID_INTENTS` | Основний намір клієнта |
| `satisfaction` | `satisfied` / `neutral` / `unsatisfied` | Реальна задоволеність клієнта |
| `quality_score` | 1–5 | Оцінка роботи агента |
| `agent_mistakes` | список з `AGENT_MISTAKES` | Конкретні помилки агента |
| `hidden_dissatisfaction` | `true` / `false` | Клієнт ввічливий, але проблема не вирішена |

### Логіка промптів

#### Intent Classification
- 6 категорій: `payment_issue`, `technical_error`, `account_access`, `tariff_question`, `refund_request`, `other`
- Розширені описи кожної категорії з конкретними прикладами
- **Правила disambiguation** для часто плутаних пар:
  - shipping options → `tariff_question`, tracking order → `other`
  - placing order → `technical_error`, complaint → `other`
  - account settings → `account_access`, invoices → `tariff_question`
  - payment failed → `payment_issue`, money back → `refund_request`

#### Satisfaction Assessment
- "satisfied" — агент відповів на питання і спробував допомогти
- "unsatisfied" — агент не допоміг, був грубий, або клієнт виразив незадоволення
- "neutral" — рідко (~15%), тільки коли результат справді неоднозначний
- Scenario-based hints для типових ситуацій

#### Quality Score (1-5)
- 5: Відмінно. Конкретні кроки для вирішення проблеми (~25%)
- 4: Добре. Корисні поради, адресує проблему (~25%)
- 3: Адекватно. Загальна, але релевантна відповідь (~15%)
- 2: Погано. Нерелевантна, занадто абстрактна відповідь (~25%)
- 1: Жахливо. Грубість, явно невірна інформація (~10%)

#### Agent Mistakes
- `ignored_question` — агент повністю проігнорував питання клієнта
- `incorrect_info` — агент надав **фактично невірну** інформацію (НЕ просто загальну чи неповну)
- `rude_tone` — нечемний, зневажливий тон
- `no_resolution` — не запропонував жодного рішення
- `unnecessary_escalation` — ескалація, коли міг вирішити сам

#### Hidden Dissatisfaction
- Надзвичайно рідко (<5% розмов)
- Тільки коли клієнт явно здається: "ладно, розберусь сам", "добре, дякую" — при невирішеній проблемі

### 5. Збереження і статистика

Результат зберігається в `data/analysis.json`. Кожен запис містить:
- `dialog` — сам діалог
- `analysis` — незалежний аналіз від LLM (5 полів)

Агреговані статистики зберігаються в `data/analysis_stats.json`:
розподіл за інтентами, satisfaction, помилками, середній quality_score.

### 6. Порівняння з генератором

Якщо в датасеті є `metadata` від генератора, аналізатор порівнює свої результати з оригінальними мітками:

| Метрика порівняння | Як рахується |
|---|---|
| Intent accuracy | exact match + per-class P/R/F1 + confusion matrix |
| Satisfaction accuracy | exact match + per-class P/R/F1 |
| Hidden dissatisfaction | P/R/F1 + FP/FN |
| Quality score | exact match, ±1 match, MAE, distribution comparison |
| Agent mistakes | binary accuracy + per-type P/R/F1 + Jaccard similarity |

## Параметри LLM

| Параметр | Значення | Опис |
|---|---|---|
| `model` | `llama3.1:8b` | Модель Ollama |
| `temperature` | `0` | Детерміністичний вихід |
| `seed` | `42` | Фіксований seed |
| `num_predict` | `2048` | Максимум токенів на виході |
| `format` | `json` | Примусовий JSON-вихід |

## Файли

| Файл | Опис |
|------|-----------|
| `analyze.py` | Точка входу: запуск процесу аналізу. |
| `src/analyzer/main.py` | Оркестрація: завантаження → батчевий аналіз → збереження → статистика → порівняння. |
| `src/analyzer/engine.py` | `DatasetAnalyzer`: two-tier LLM-запити, парсинг JSON, валідація і мерж результатів. |
| `src/analyzer/prompts.py` | Шаблони промптів: Tier 1 (intent+satisfaction), Tier 2 (quality+mistakes+HD), single-dialog fallback. |
| `src/config/constants.py` | Константи: `VALID_INTENTS`, `AGENT_MISTAKES`, `TEMPLATE_REPLACEMENTS`, `SEED`. |
| `src/config/logger.py` | Налаштування логування для відстеження процесу аналізу. |

## Конфігурація та налаштування

- `src/config/constants.py`: зміна `VALID_INTENTS` або `AGENT_MISTAKES` вплине на валідацію.
- `src/analyzer/prompts.py`: основне місце для налаштування логіки — описи інтентів, рубрика якості, правила disambiguation.
- `src/analyzer/engine.py`: `tier1_batch_size=15`, `tier2_batch_size=10` — регулюють навантаження на LLM.

### Робота в Docker
Якщо ви запускаєте аналіз у Docker-контейнері, переконайтеся, що `OLLAMA_HOST` вказано правильно (за замовчуванням у `docker-compose.yml` це `http://ollama:11434`).
