import importlib
import os


def test_bot_admin_id_parsing(monkeypatch):
    # simulate inline comment in .env
    monkeypatch.setenv("BOT_ADMIN_ID", "662750197          # numeric user id (желательно)")
    import bot as b
    importlib.reload(b)
    assert isinstance(b.BOT_ADMIN_ID, int)
    assert b.BOT_ADMIN_ID == 662750197
