import asyncio
import os

import bot


def test_notify_admin_on_start(monkeypatch):
    calls = []

    async def fake_send_message(chat_id, text=None, **kwargs):
        calls.append((chat_id, text))

    monkeypatch.setenv("BOT_ADMIN_ID", "12345")
    bot.BOT_ADMIN_ID = 12345
    monkeypatch.setattr(bot, "bot", bot.bot)
    monkeypatch.setattr(bot.bot, "send_message", fake_send_message)

    asyncio.run(bot.notify_admin_startup())

    assert any(call[0] == 12345 and "Bot started" in (call[1] or "") for call in calls)
