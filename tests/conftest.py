import os
import pytest
import sys
import os

# Ensure project root is importable during tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import bot as _bot


class FakeBot:
    async def edit_message_text(self, *args, **kwargs):
        return None

    async def send_message(self, *args, **kwargs):
        return None

    async def set_my_commands(self, *args, **kwargs):
        return None


@pytest.fixture(autouse=True)
def disable_real_bot(monkeypatch):
    # Ensure tests don't use a real BOT_TOKEN or network calls
    # provide a dummy token so module import does not fail and aiogram accepts it
    monkeypatch.setenv("BOT_TOKEN", "123:FAKE_TOKEN")
    # replace bot.bot with a fake instance
    monkeypatch.setattr(_bot, "bot", FakeBot())
    yield
