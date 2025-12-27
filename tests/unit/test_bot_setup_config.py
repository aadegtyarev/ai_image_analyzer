import importlib


def test_setup_config_attaches_cfg(monkeypatch):
    # ensure load_config returns an object with expected attributes
    class C:
        api_key = "x"
        base_url = "http://example"

    monkeypatch.setattr("ai_image_analyzer.load_config", lambda: C())

    m = importlib.import_module("bot.main")
    cfg = m.setup_config()
    assert cfg is not None
    import bot as pkg
    assert getattr(pkg, "cfg", None) is not None
    # pkg should also receive BOT_TOKEN from environment when setup_config is called
    import os
    os.environ["BOT_TOKEN"] = "tkn"
    cfg2 = m.setup_config()
    assert getattr(pkg, "BOT_TOKEN", None) == "tkn"


def test_bot_setup_tolerant_bot_token_parsing(monkeypatch, tmp_path):
    # Create a malformed .env where BOT_TOKEN and BOT_ADMIN_ID are on same line
    env_file = tmp_path / ".env"
    env_file.write_text("BOT_TOKEN=abc123   BOT_ADMIN_ID=555\n")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    # point find_dotenv to this file
    import ai_image_analyzer.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "find_dotenv", lambda *a, **k: str(env_file))
    m = importlib.import_module("bot.main")
    cfg = m.setup_config()
    import bot as pkg2
    assert pkg2.BOT_TOKEN == "abc123"
