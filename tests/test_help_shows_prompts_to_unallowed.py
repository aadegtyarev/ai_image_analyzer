import asyncio
from types import SimpleNamespace

import bot


class DummyMsg:
    def __init__(self):
        self.message_id = 1
        self.chat = SimpleNamespace(type="private")
        self.from_user = SimpleNamespace(id=999999)

    async def answer(self, text=None, parse_mode=None):
        return self


def test_help_shows_prompts_to_unallowed(monkeypatch):
    msg = DummyMsg()

    captured = {}

    async def fake_send_response(m, text, filename_prefix=None):
        captured["text"] = text

    monkeypatch.setattr(bot, "send_response", fake_send_response)

    # Ensure user not in allowed list
    if 999999 in bot.users_db.get("enabled", []):
        bot.users_db["enabled"].remove(999999)

    asyncio.run(bot.handle_help(msg))

    assert "Команды:" in captured.get("text", ""), "help should show commands"
    assert "Прмты" not in captured.get("text", "")  # admin-only prompts header shouldn't appear
