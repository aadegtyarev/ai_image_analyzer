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
