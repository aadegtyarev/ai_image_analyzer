import importlib


def test_setup_config_attaches_cfg(monkeypatch):
    # ensure load_config returns an object with expected attributes
    class C:
        api_key = "x"
        base_url = "http://example"

    monkeypatch.setattr("ai_image_analyzer.config.load_dotenv", lambda *a, **k: True)
    monkeypatch.setattr("ai_image_analyzer.config.find_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("ai_image_analyzer.config.load_config", lambda: C())

    m = importlib.import_module("bot.main")
    cfg = m.setup_config()
    assert cfg is not None
    import bot as pkg
    assert getattr(pkg, "cfg", None) is not None
