import time
from types import SimpleNamespace
import asyncio
import os

import bot


class DummyMsgWithAnswer:
    def __init__(self):
        self.message_id = 99
        self.chat = type("C", (), {"id": 12345})()
        self.from_user = type("U", (), {"id": 5555})()

    async def answer(self, text=None, parse_mode=None):
        # Return a message-like object
        return self


def test_edit_failure_no_notify_admin(monkeypatch, tmp_path):
    mgid = "mg_edit_admin"
    imgs = [(b"a", 100, 100, "square")]
    reply = DummyMsgWithAnswer()

    bot.MEDIA_CONTEXTS[mgid] = {
        "images": imgs,
        "system_prompt": "SP",
        "prompt_label": "p",
        "use_text_override": False,
        "user_text": None,
        "force_collage": True,
        "reply_msg": reply,
        "ts": time.time() - 10,
    }

    # make_collage stub
    def fake_make_collage(images, max_size, jpeg_quality):
        fnames = ["image_1.jpg"]
        return (b"collage", fnames)

    monkeypatch.setattr(bot, "make_collage", fake_make_collage)

    # edit_message_text will raise
    async def fake_edit_message_text(*args, **kwargs):
        raise RuntimeError("edit failed")

    monkeypatch.setattr(bot.bot, "edit_message_text", fake_edit_message_text)

    # stub call_model_with_image
    def fake_call_model_with_image(cfg, data, system_prompt=None, user_text=None, quiet=False, image_meta=None):
        return ("RESULT", None)

    monkeypatch.setattr(bot, "call_model_with_image", fake_call_model_with_image)

    # capture admin notifications
    admin_calls = []

    async def fake_send_message(chat_id, text=None, **kwargs):
        admin_calls.append((chat_id, text))

    monkeypatch.setattr(bot.bot, "send_message", fake_send_message)

    # assign admin id in module under test
    bot.BOT_ADMIN_ID = 99999

    # disable admin notifications via env
    os.environ["NOTIFY_ON_EDIT_FAILURE"] = "0"

    cfg = SimpleNamespace(collage_max_size=1024, collage_quality=85)
    bot.GROUP_WAIT = 0

    asyncio.run(bot._process_media_group(mgid, cfg))

    # ensure admin was NOT notified
    assert len(admin_calls) == 0
