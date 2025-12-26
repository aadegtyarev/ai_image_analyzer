# Архитектура и план рефакторинга

Кратко: цель — разделить проект на библиотеку (ядро) и тонкий адаптер бота, убрать CLI как центр, сделать провайдеры/сервис для взаимодействия с LLM/vision API и покрыть всё тестами.

## Целевая архитектура

ai_image_analyzer/ (пакет — ядро)
- `config.py` — загрузка `.env`, `Settings`, `load_config()`, `read_prompt_file()`
- `image_processing.py` — ресайз, коллаж, `build_collage_system_prompt()`
- `image_io.py` — совместимые функции `load_and_resize_image`, `resize_image_from_bytes`
- `model_client.py` — интерфейс к внешним моделям (`call_model_with_image`, `call_model_with_text_only`, `check_balance`) и провайдеры
- `json_api.py` — `handle_json_request()` для внешних клиентов/тестов
- `__init__.py` — фасад с экспортами для совместимости

bot/ (тонкий адаптер Telegram)
- `config.py` — BOT_TOKEN, пути, флаги
- `users_store.py` — работа с `db/users.json` и статистикой
- `formatting.py` — `simple_markdown_to_html`, `send_response`
- `media_group.py` — `MEDIA_CONTEXTS`, `_process_media_group`, TTL
- `handlers.py` — регистрация роутера и обработчики команд/фото/альбомов
- `main.py` — создание `Bot/Dispatcher/Router` и запуск

CLI scripts (тонкие обёртки)
- `ai_image_analyzer.py` — thin CLI (опционально)
- `bot.py` — thin runner для `bot.main.run()`

## Последовательность работ (конкретные шаги)

1. Пакет `ai_image_analyzer` (ядро)
   - Вынести/реализовать: `config`, `image_processing`, `image_io`, `model_client`, `json_api`, `__init__`
   - Сохранить сигнатуры и имена, используемые в тестах/боте
   - Добавить unit-тесты для каждого модуля

2. Провайдеры и сервисы
   - Определить `ModelProvider` protocol и `OpenAIProvider` (реализация через SDK)
   - Добавить `AnalyzerService`, который делегирует провайдерам
   - Покрыть unit и интеграционными тестами (mock SDK, mock HTTP)

3. Рефакторинг бота в `bot/`
   - Перенести код `bot.py` в модули, минимизировать логику в каждом модуле
   - Хендлеры должны использовать API ядра (`call_model_with_image` и т.д.)
   - Добавить тесты для медиагрупп/поведения бота

4. Документация и CI
   - Обновить README, CHANGELOG, добавить примеры использования и диалогов
   - Настроить CI (pytest) для прогонки unit и интеграционных тестов

5. Дополнения (по желанию)
   - Поддержка дополнительных провайдеров (локальные модели, другие API)
   - Улучшение логирования/метрик (usage, cost logging)
   - Интеграционные тесты с реальными ключами как опция (skip в CI)

---

Файл создан на основании обсуждения и согласованного ТЗ. Ведите счётчик прогресса через TODOs в проекте (файл `TODO.md` или task list в issues). 
