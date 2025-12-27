Legacy / deprecated files
=========================

This directory contains legacy monolithic scripts that were replaced by the
modular `ai_image_analyzer` package and the `bot/` package.

Files:
- `ai_image_analyzer.py` — original CLI (deprecated).
- `bot.py` — original monolithic bot runner (deprecated). A thin shim `bot.py`
  exists at the repository root that delegates to the modular `bot` package.

These files were preserved for reference. If you need to restore them, a
backup branch was created (`backup/legacy-save-YYYYMMDD_HHMMSS`).
