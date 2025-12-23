# Changelog

All notable changes to this project are documented in this file.

## Unreleased

- Fix: media-group prompt caching — the bot now caches system/user prompt per Telegram media_group_id so all images in an album use the same prompt (configurable TTL via `MEDIA_CONTEXT_TTL`). ✅
- Add: `[MEDIA_DEBUG]` log lines when a media-group context is set or expired (visible when `DEBUG=1`). ✅
- Fix: UnboundLocalError when caching prompt without override text — `user_text` is now initialized and safe. ✅
- Change: removed on-disk payload dumping (`DUMP_PAYLOADS`) — debug is now logged to stderr only (simpler, safer). ✅
- Change: unified debug flags — use `DEBUG=1` (general) and `IMAGE_DEBUG=1` (image details). ✅
- Add: unit tests for media-group caching and logging; updated/simplified tests around prompts and debug. ✅
- Docs: updated `BOT.md`, `README.md` and `env.example` to reflect changes above. ✅

---

For details see recent commits on `main`.
