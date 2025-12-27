import os
from pathlib import Path

from ai_image_analyzer import load_config


def test_load_config_finds_env_in_parent(tmp_path, monkeypatch):
    # create project root with .env
    root = tmp_path / "proj"
    sub = root / "a" / "b"
    sub.mkdir(parents=True)
    env_file = root / ".env"
    env_file.write_text("OPENAI_API_KEY=parent-key\nOPENAI_MODEL=parent-model\nBOT_TOKEN=abc123\n")

    # simulate running from nested subfolder
    monkeypatch.chdir(sub)

    # Ensure find_dotenv returns empty to exercise our fallback search
    import ai_image_analyzer.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "find_dotenv", lambda *a, **k: "")

    cfg = load_config()
    assert cfg.api_key == "parent-key"
    assert cfg.model == "parent-model"
    assert os.environ.get("BOT_TOKEN") == "abc123"
