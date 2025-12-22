## Быстрое введение — что делает проект

Python-утилита для анализа фотографий через OpenAI‑совместимые vision-модели.

- CLI: `python ai_image_analyzer.py [OPTIONS] [images...]` — один запрос на картинку, несколько → коллаж, `--per-image` → отдельный запрос для каждого файла.
- Модульный JSON-API: `handle_json_request(dict)` / `handle_json_request_str(str)` — удобно для ботов (Telegram и т.д.).

## Важные архитектурные детали (быстро)

- Входные данные: файлы с диска (`path`) или байты (`data_b64`) — код обрабатывает оба варианта (`load_and_resize_image`, `resize_image_from_bytes`).
- Изображения ресайзятся до `IMAGE_MAX_SIZE` (по длинной стороне), по умолчанию 1024px; качество JPEG задаётся переменной `IMAGE_QUALITY`.
- Коллаж: строится простая квадратная сетка (автоматическое N×M) в `make_collage`. Порядок ячеек — слева направо, сверху вниз.
- Ограничение: максимум 10 изображений в одном запросе (CLI и JSON-api).
- Промты лежат в `prompts/`. `PROMPT_FILE` (или `--PROMPT_FILE`) используется как **system prompt**; при передаче `-t/override_text` он **игнорируется** и текст идёт как `user`.
- Модели и токены: добавлена переменная `OPENAI_MAX_TOKENS` (опционально, по умолчанию не ограничено).
- Баланс и биллинг: добавлены следующие возможности:
  - `OPENAI_BALANCE_THRESHOLD` (env) и CLI `--check-balance` — запрос баланса (`/v1/balance`) и предупреждение, если баланс ниже порога. JSON API принимает `check_balance: true`.
  - Поля `usage` (prompt_tokens/completion_tokens/total_tokens/total_cost) всегда возвращаются в поле `usage` в каждом результате и выводятся в CLI как `USAGE:` (если не `-q`). `total_cost` округляется до 3 знаков после запятой.
  - Эти данные возвращаются «как есть» от провайдера (числа токенов/стоимости), но total_cost округлён.

## Что нужно знать при изменении кода (полезные точки входа)

- CLI: `main()` и `parse_args()` — управление флагами и перевод CLI-флагов в `os.environ`.
- JSON API: `handle_json_request()` → `analyze_json()` — логика пер-изображение/коллаж/только-текст идентична CLI, но **не пишет на диск**.
- Вызовы моделей: `call_model_with_image()` и `call_model_with_text_only()` — правьте их если меняется API SDK (обёртка `build_client()` собирает `OpenAI(api_key=..., base_url=...)`).
- Чтение промта: `read_prompt_file()`; при отсутствии `PROMPT_FILE` скрипт может завершать работу — учтите это в тестах/изменениях.

## Проектные конвенции и тонкие вещи, которые нужно соблюдать

- Output filenames: одноизображение → `*_analyse.md`, группа → `group_YYYYMMDD_HHMMSS_analyse.txt` (используйте ту же форму при изменении сохранения).
- Поведение `-t/override_text`: если передан текст вместе с изображениями — результат печатается в stdout и **не** сохраняется в файл.
- JSON-API возвращает структуру `{ok, mode, results, text_only, errors}` — сохраняйте совместимость при изменениях.
- Логи пишутся в stderr через `log(msg, quiet)`; `-q/--quiet` подавляет подробные сообщения.

## Зависимости и локальная проверка

- Требуется Python 3.9+. Установите `pip install -r requirements.txt` (`openai`, `Pillow`, `python-dotenv`).
- Для быстрой проверки (без файлов):
  - Текстовый запрос: `python ai_image_analyzer.py -t "тест"` (но нужен `OPENAI_API_KEY` и пакет `openai`).
  - Для модульной проверки используйте `handle_json_request({'action':'analyze', 'override_text':'...', 'images': []})`.

## Частые места для правок и тестов (рекомендации для PR)

- Если меняете поведение ресайза/коллажа — добавьте проверку размеров и порядок имен файлов (см. `make_collage` и `build_collage_system_prompt`).
- При изменении формата ответов тщательно обновите `README.md` и JSON API-ответ в `analyze_json()`.
- Добавляйте unit-тесты для: resize (ввод bytes и path), collage layout (порядок и размеры тайлов), поведение per-image vs collage и обработку ошибок (некорректный base64, отсутствующий prompt).

## Быстрые примеры (копировать/выполнить)

- CLI (одна картинка → файл):
  - `python ai_image_analyzer.py images/photo.jpg`
- CLI (несколько → коллаж + group-файл):
  - `python ai_image_analyzer.py "images/*.jpg"`
- JSON (пер-изображение из байтов):
  - `handle_json_request({"action":"analyze","per_image":true,"images":[{"name":"a.jpg","data_b64":"..."}]})`

---

Если нужно — могу сократить или дополнить инструкции примерами кода/юнит-тестами; скажите, какую часть хочется подробнее описать. ✅
